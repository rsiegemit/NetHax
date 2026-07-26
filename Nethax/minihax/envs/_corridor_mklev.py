"""Faithful N-room Corridor mklev engine (rect pool + create_room + STAIR +
makecorridors + find_branch_room), consuming state.vendor_rng. Ported from the
validated standalone simulator .test_runs/_corr_sim.py; reproduces vendor
NetHack corridor level-gen bit-exact for the reset observation. Eager Python.
"""
import numpy as np
from Nethax.nethax import vendor_rng as VR

COLNO, ROWNO = 80, 21
XLIM, YLIM = 4, 3

# ---- terrain typ codes we care about (vendor rm.typ) ----
STONE = 0
VWALL, HWALL = 1, 2
# Use sentinels; we only need occupancy + ROOM/CORR/DOOR/wall classification.
ROOM_T = 100
CORR_T = 101
DOOR_T = 102
SCORR_T = 103
SDOOR_T = 104
TLCORNER=110; TRCORNER=111; BLCORNER=112; BRCORNER=113

def IS_WALL(t): return t in (VWALL, HWALL, TLCORNER,TRCORNER,BLCORNER,BRCORNER)
def IS_DOOR(t): return t in (DOOR_T,)

OROOM_RT = 0   # OROOM rtype


TRACE_TAG = [""]
class RNG:
    def __init__(self, state):
        self.s = state
        self.log = []
        self.tags = []
    def rn2(self, n):
        self.s, r = VR.rn2_jax(self.s, int(n))
        r = int(r)
        self.log.append((int(n), r))
        self.tags.append(TRACE_TAG[0])
        return r
    def rn1(self, x, base):
        return self.rn2(x) + base
    def rnd(self, n):
        self.s, r = VR.rnd_jax(self.s, int(n))
        return int(r)


class Level:
    def __init__(self):
        self.typ = np.zeros((COLNO, ROWNO), dtype=np.int32)  # [x][y]
        self.horiz = np.zeros((COLNO, ROWNO), dtype=np.int32)
        self.doormask = np.zeros((COLNO, ROWNO), dtype=np.int32)
        self.rooms = []   # list of dict lx,ly,hx,hy,rtype,rlit
        self.smeq = []
        self.doorct_total = 0
        self.doors = []
        self.upstairs_room = None
        self.dnstairs_room = None
        self.dnstairs_cell = None


# ---------------- rect pool (faithful rect.c) ----------------
class Rect:
    __slots__=("lx","ly","hx","hy")
    def __init__(self,lx,ly,hx,hy): self.lx=lx; self.ly=ly; self.hx=hx; self.hy=hy
    def copy(self): return Rect(self.lx,self.ly,self.hx,self.hy)
    def eq(self,o): return self.lx==o.lx and self.ly==o.ly and self.hx==o.hx and self.hy==o.hy

class RectPool:
    def __init__(self):
        self.rect=[Rect(0,0,COLNO-1,ROWNO-1)]
    def rnd_rect(self, rng):
        if not self.rect: return None
        return self.rect[rng.rn2(len(self.rect))]
    def get_rect_ind(self, r):
        for i,rp in enumerate(self.rect):
            if r.lx==rp.lx and r.ly==rp.ly and r.hx==rp.hx and r.hy==rp.hy:
                return i
        return -1
    def get_rect(self, r):
        for rp in self.rect:
            if r.lx>=rp.lx and r.ly>=rp.ly and r.hx<=rp.hx and r.hy<=rp.hy:
                return rp
        return None
    def remove_rect(self, r):
        ind=self.get_rect_ind(r)
        if ind>=0:
            self.rect[ind]=self.rect[-1]
            self.rect.pop()
    def add_rect(self, r):
        if len(self.rect)>=50: return
        if self.get_rect(r) is not None: return
        self.rect.append(r.copy())
    def split_rects(self, r1, r2):
        old=r1.copy()
        self.remove_rect(r1)
        for i in range(len(self.rect)-1,-1,-1):
            inter=intersect(self.rect[i], r2)
            if inter is not None:
                self.split_rects(self.rect[i], inter)
        if r2.ly-old.ly-1 > (2*YLIM if old.hy<ROWNO-1 else YLIM+1)+4:
            r=old.copy(); r.hy=r2.ly-2; self.add_rect(r)
        if r2.lx-old.lx-1 > (2*XLIM if old.hx<COLNO-1 else XLIM+1)+4:
            r=old.copy(); r.hx=r2.lx-2; self.add_rect(r)
        if old.hy-r2.hy-1 > (2*YLIM if old.ly>0 else YLIM+1)+4:
            r=old.copy(); r.ly=r2.hy+2; self.add_rect(r)
        if old.hx-r2.hx-1 > (2*XLIM if old.lx>0 else XLIM+1)+4:
            r=old.copy(); r.lx=r2.hx+2; self.add_rect(r)

def intersect(r1, r2):
    if r2.lx>r1.hx or r2.ly>r1.hy or r2.hx<r1.lx or r2.hy<r1.ly:
        return None
    return Rect(max(r2.lx,r1.lx), max(r2.ly,r1.ly), min(r2.hx,r1.hx), min(r2.hy,r1.hy))


# ---------------- add_room / do_room_or_subroom ----------------
def add_room(lev, lowx, lowy, hix, hiy, lit, rtype):
    # do_room_or_subroom is_room=TRUE, special=FALSE
    if lowx==0: lowx=1
    if lowy==0: lowy=1
    if hix>=COLNO-1: hix=COLNO-2
    if hiy>=ROWNO-1: hiy=ROWNO-2
    T=lev.typ
    # top/bottom HWALL
    for x in range(lowx-1, hix+2):
        for y in range(lowy-1, hiy+2, (hiy-lowy+2)):
            T[x][y]=HWALL; lev.horiz[x][y]=1
    # left/right VWALL
    for x in range(lowx-1, hix+2, (hix-lowx+2)):
        for y in range(lowy, hiy+1):
            T[x][y]=VWALL; lev.horiz[x][y]=0
    # floor
    for x in range(lowx, hix+1):
        for y in range(lowy, hiy+1):
            T[x][y]=ROOM_T
    # corners
    T[lowx-1][lowy-1]=TLCORNER
    T[hix+1][lowy-1]=TRCORNER
    T[lowx-1][hiy+1]=BLCORNER
    T[hix+1][hiy+1]=BRCORNER
    room=dict(lx=lowx, ly=lowy, hx=hix, hy=hiy, rtype=rtype, rlit=lit)
    lev.rooms.append(room)
    lev.smeq.append(len(lev.rooms)-1)
    return room


# ---------------- check_room ----------------
def check_room(lev, lowx, ddx, lowy, ddy, rng):
    xlim, ylim = XLIM, YLIM
    hix = lowx + ddx
    hiy = lowy + ddy
    if lowx < 3: lowx = 3
    if lowy < 2: lowy = 2
    if hix > COLNO-3: hix = COLNO-3
    if hiy > ROWNO-3: hiy = ROWNO-3
    while True:
        if hix <= lowx or hiy <= lowy:
            return None
        restart=False
        for x in range(lowx-xlim, hix+xlim+1):
            if x<=0 or x>=COLNO: continue
            y = lowy-ylim
            ymax = hiy+ylim
            if y<0: y=0
            if ymax>=ROWNO: ymax=ROWNO-1
            hit=False
            for yy in range(y, ymax+1):
                if lev.typ[x][yy]:
                    if rng.rn2(3)==0:
                        return None
                    if x < lowx: lowx = x+xlim+1
                    else: hix = x-xlim-1
                    if yy < lowy: lowy = yy+ylim+1
                    else: hiy = yy-ylim-1
                    hit=True
                    break
            if hit:
                restart=True
                break
        if restart:
            continue
        ddx = hix-lowx
        ddy = hiy-lowy
        return (lowx, ddx, lowy, ddy)


# ---------------- build_room wrapper (chance roll) ----------------
def build_room(lev, pool, rlit, rng):
    # r->chance != 0 for "ordinary" ROOM -> draw rn2(100). rtype resolves to
    # OROOM regardless (r->rtype == OROOM for ordinary), so value is unused.
    rng.rn2(100)
    return create_room_random(lev, pool, OROOM_RT, rlit, rng)


# ---------------- create_room (fully random branch) ----------------
def create_room_random(lev, pool, rtype, rlit, rng):
    trycnt=0
    r1=None
    xabs=yabs=wtmp=htmp=0
    r2=None
    while True:
        r1 = pool.rnd_rect(rng)
        if r1 is None:
            return False
        hx,hy,lx,ly = r1.hx, r1.hy, r1.lx, r1.ly
        dx = 2 + rng.rn2(12 if (hx-lx>28) else 8)
        dy = 2 + rng.rn2(4)
        if dx*dy > 50:
            dy = 50//dx
        xborder = 2*XLIM if (lx>0 and hx<COLNO-1) else XLIM+1
        yborder = 2*YLIM if (ly>0 and hy<ROWNO-1) else YLIM+1
        if hx-lx < dx+3+xborder or hy-ly < dy+3+yborder:
            r1=None
            trycnt+=1
            if trycnt>100: break
            continue
        xabs = lx + (XLIM if lx>0 else 3) + rng.rn2(hx-(lx if lx>0 else 3)-dx-xborder+1)
        yabs = ly + (YLIM if ly>0 else 2) + rng.rn2(hy-(ly if ly>0 else 2)-dy-yborder+1)
        nroom = len(lev.rooms)
        if ly==0 and hy>=ROWNO-1 and (nroom==0 or rng.rn2(nroom)==0) and (yabs+dy>ROWNO//2):
            yabs = rng.rn1(3,2)
            if nroom<4 and dy>1:
                dy-=1
        chk = check_room(lev, xabs, dx, yabs, dy, rng)
        if chk is None:
            r1=None
            trycnt+=1
            if trycnt>100: break
            continue
        xabs, dx, yabs, dy = chk
        wtmp=dx+1; htmp=dy+1
        r2=Rect(xabs-1, yabs-1, xabs+wtmp, yabs+htmp)
        break
    if r1 is None:
        return False
    pool.split_rects(r1, r2)
    add_room(lev, xabs, yabs, xabs+wtmp-1, yabs+htmp-1, rlit, rtype)
    return True


# ---------------- STAIR (get_location random, DRY) ----------------
def spo_stair(lev, croom, up, rng):
    # get_location_coord random -> somexy(croom): somex=rn1(hx-lx+1,lx), somey
    for _ in range(100):
        x = rng.rn1(croom['hx']-croom['lx']+1, croom['lx'])
        y = rng.rn1(croom['hy']-croom['ly']+1, croom['ly'])
        typ = lev.typ[x][y]
        # is_ok_location DRY: ROOM/CORR accepted
        if typ in (ROOM_T, CORR_T):
            break
    if up:
        # Dlvl-1 up: mkstairs early-returns; no stair placed; upstairs_room stays NULL
        return
    else:
        lev.dnstairs_room = croom
        lev.dnstairs_cell = (x, y)
        # place down stair (mark a non-room typ so it doesn't affect corridors much)
        # keep typ ROOM for simplicity; stair is stripped by minihack anyway.


# ---------------- door helpers ----------------
def bydoor(lev, x, y):
    for (dx,dy) in ((1,0),(-1,0),(0,1),(0,-1)):
        xx,yy=x+dx,y+dy
        if 0<=xx<COLNO and 0<=yy<ROWNO:
            t=lev.typ[xx][yy]
            if t in (SDOOR_T, DOOR_T) or (t==CORR_T):
                # doorindex>0 check omitted; vendor bydoor checks IS_DOOR||typ==SDOOR||SCORR
                pass
    # faithful bydoor: SDOOR/DOOR/SCORR adjacency (incl. current cell? no)
    for (dx,dy) in ((1,0),(-1,0),(0,1),(0,-1),(0,0)):
        xx,yy=x+dx,y+dy
        if 0<=xx<COLNO and 0<=yy<ROWNO:
            t=lev.typ[xx][yy]
            if t in (SDOOR_T, DOOR_T, SCORR_T):
                return True
    return False

def okdoor(lev, x, y):
    t=lev.typ[x][y]
    return (t in (HWALL, VWALL)) and (not bydoor(lev, x, y))

def dosdoor(lev, x, y, typ, rng):
    # not shop. if not wall -> DOOR
    if not IS_WALL(lev.typ[x][y]):
        typ=DOOR_T
    lev.typ[x][y]=typ
    if typ==DOOR_T:
        if rng.rn2(3)==0:
            if rng.rn2(5)==0:
                lev.doormask[x][y]=1  # ISOPEN
            elif rng.rn2(6)==0:
                lev.doormask[x][y]=4  # LOCKED
            else:
                lev.doormask[x][y]=2  # CLOSED
            # level_difficulty >=5? corridor depth 1 -> <5, skip D_TRAPPED rn2(25)
        else:
            lev.doormask[x][y]=0  # NODOOR
    else:
        # SDOOR (secret door): vendor dosdoor mklev.c else-branch draws rn2(5)
        # (if (!rn2(5)) D_LOCKED else D_CLOSED).  Renders as wall regardless, but
        # the draw MUST fire to keep the ISAAC64 stream aligned.
        if rng.rn2(5)==0:
            lev.doormask[x][y]=4  # LOCKED
        else:
            lev.doormask[x][y]=2  # CLOSED

def dodoor(lev, x, y, rng):
    TRACE_TAG[0]="dodoor8"
    dosdoor(lev, x, y, DOOR_T if rng.rn2(8) else SDOOR_T, rng)


# ---------------- finddpos ----------------
def finddpos(lev, xl, yl, xh, yh, rng):
    TRACE_TAG[0]="finddpos"
    x = rng.rn1(xh-xl+1, xl)
    y = rng.rn1(yh-yl+1, yl)
    if okdoor(lev, x, y):
        return (x,y)
    for x in range(xl, xh+1):
        for y in range(yl, yh+1):
            if okdoor(lev, x, y):
                return (x,y)
    for x in range(xl, xh+1):
        for y in range(yl, yh+1):
            if IS_DOOR(lev.typ[x][y]) or lev.typ[x][y]==SDOOR_T:
                return (x,y)
    return (xl, yh)


# ---------------- dig_corridor ----------------
def dig_corridor(lev, org, dest, nxcor, ftyp, btyp, rng):
    dx=dy=0
    xx,yy=org; tx,ty=dest
    if xx<=0 or yy<=0 or tx<=0 or ty<=0 or xx>COLNO-1 or tx>COLNO-1 or yy>ROWNO-1 or ty>ROWNO-1:
        return False
    if tx>xx: dx=1
    elif ty>yy: dy=1
    elif tx<xx: dx=-1
    else: dy=-1
    xx-=dx; yy-=dy
    cct=0
    while xx!=tx or yy!=ty:
        cct+=1
        if cct>500:
            return False
        if nxcor:
            TRACE_TAG[0]="dig35"
            if rng.rn2(35)==0:
                return False
        xx+=dx; yy+=dy
        if xx>=COLNO-1 or xx<=0 or yy<=0 or yy>=ROWNO-1:
            return False
        t=lev.typ[xx][yy]
        if t==btyp:
            TRACE_TAG[0]="digcarve"
            if ftyp!=CORR_T or rng.rn2(100):
                lev.typ[xx][yy]=ftyp
                if nxcor:
                    TRACE_TAG[0]="digboulder"
                    if rng.rn2(50)==0:
                        pass  # boulder
            else:
                lev.typ[xx][yy]=SCORR_T
        elif t!=ftyp and t!=SCORR_T:
            return False
        dix=abs(xx-tx); diy=abs(yy-ty)
        TRACE_TAG[0]="digbend"
        if (dix>diy) and diy and rng.rn2(dix-diy+1)==0:
            dix=0
        elif (diy>dix) and dix and rng.rn2(diy-dix+1)==0:
            diy=0
        if dy and dix>diy:
            ddx = -1 if xx>tx else 1
            t2=lev.typ[xx+ddx][yy]
            if t2 in (btyp,ftyp,SCORR_T):
                dx=ddx; dy=0; continue
        elif dx and diy>dix:
            ddy = -1 if yy>ty else 1
            t2=lev.typ[xx][yy+ddy]
            if t2 in (btyp,ftyp,SCORR_T):
                dy=ddy; dx=0; continue
        t3=lev.typ[xx+dx][yy+dy]
        if t3 in (btyp,ftyp,SCORR_T):
            continue
        if dx:
            dx=0; dy = -1 if ty<yy else 1
        else:
            dy=0; dx = -1 if tx<xx else 1
        t4=lev.typ[xx+dx][yy+dy]
        if t4 in (btyp,ftyp,SCORR_T):
            continue
        dy=-dy; dx=-dx
    return True


# ---------------- join ----------------
def join(lev, a, b, nxcor, rng):
    croom=lev.rooms[a]; troom=lev.rooms[b]
    if troom['hx']<0 or croom['hx']<0:
        return
    if troom['lx']>croom['hx']:
        dx=1; dy=0
        xx=croom['hx']+1; tx=troom['lx']-1
        cc=finddpos(lev, xx, croom['ly'], xx, croom['hy'], rng)
        tt=finddpos(lev, tx, troom['ly'], tx, troom['hy'], rng)
    elif troom['hy']<croom['ly']:
        dy=-1; dx=0
        yy=croom['ly']-1
        cc=finddpos(lev, croom['lx'], yy, croom['hx'], yy, rng)
        ty=troom['hy']+1
        tt=finddpos(lev, troom['lx'], ty, troom['hx'], ty, rng)
    elif troom['hx']<croom['lx']:
        dx=-1; dy=0
        xx=croom['lx']-1; tx=troom['hx']+1
        cc=finddpos(lev, xx, croom['ly'], xx, croom['hy'], rng)
        tt=finddpos(lev, tx, troom['ly'], tx, troom['hy'], rng)
    else:
        dy=1; dx=0
        yy=croom['hy']+1; ty=troom['ly']-1
        cc=finddpos(lev, croom['lx'], yy, croom['hx'], yy, rng)
        tt=finddpos(lev, troom['lx'], ty, troom['hx'], ty, rng)
    xx,yy=cc
    tx=tt[0]-dx; ty=tt[1]-dy
    if nxcor and lev.typ[xx+dx][yy+dy]:
        return
    if okdoor(lev, xx, yy) or (not nxcor):
        dodoor(lev, xx, yy, rng)
    org=(xx+dx, yy+dy)
    dest=(tx, ty)
    if not dig_corridor(lev, org, dest, nxcor, CORR_T, STONE, rng):
        return
    if okdoor(lev, tt[0], tt[1]) or (not nxcor):
        dodoor(lev, tt[0], tt[1], rng)
    if lev.smeq[a]<lev.smeq[b]:
        lev.smeq[b]=lev.smeq[a]
    else:
        lev.smeq[a]=lev.smeq[b]


def makecorridors(lev, rng):
    nroom=len(lev.rooms)
    a=0
    while a<nroom-1:
        join(lev, a, a+1, False, rng)
        TRACE_TAG[0]="mkcorr_rn50"
        if rng.rn2(50)==0:
            break
        a+=1
    for a in range(nroom-2):
        if lev.smeq[a]!=lev.smeq[a+2]:
            join(lev, a, a+2, False, rng)
    any_=True
    a=0
    while any_ and a<nroom:
        any_=False
        for b in range(nroom):
            if lev.smeq[a]!=lev.smeq[b]:
                join(lev, a, b, False, rng)
                any_=True
        a+=1
    if nroom>2:
        i=rng.rn2(nroom)+4
        while i:
            a=rng.rn2(nroom)
            b=rng.rn2(nroom-2)
            if b>=a: b+=2
            join(lev, a, b, True, rng)
            i-=1


# ---------------- find_branch_room (hero) ----------------
def occupied(lev, x, y):
    return False  # no traps/furniture/lava on corridor level start room

def find_branch_room(lev, rng):
    nroom=len(lev.rooms)
    if nroom>2:
        tryct=0
        while True:
            idx=rng.rn2(nroom)
            croom=lev.rooms[idx]
            reject = (croom is lev.dnstairs_room or croom is lev.upstairs_room or croom['rtype']!=OROOM_RT)
            tryct+=1
            if not (reject and tryct<100):
                break
    else:
        idx=rng.rn2(nroom)
        croom=lev.rooms[idx]
    while True:
        x=rng.rn1(croom['hx']-croom['lx']+1, croom['lx'])
        y=rng.rn1(croom['hy']-croom['ly']+1, croom['ly'])
        if not occupied(lev,x,y) and lev.typ[x][y] in (CORR_T, ROOM_T):
            break
    return croom, (x, y)


def simulate(vrng_state, n_rooms):
    rng=RNG(vrng_state)
    lev=Level()
    pool=RectPool()
    # mklev prologue: rn2(3), rn2(2)
    rng.rn2(3); rng.rn2(2)
    # ROOM1 up-stair
    build_room(lev, pool, 1, rng)
    spo_stair(lev, lev.rooms[-1], up=1, rng=rng)
    # ROOM2 down-stair
    build_room(lev, pool, 1, rng)
    spo_stair(lev, lev.rooms[-1], up=0, rng=rng)
    # ROOM3.. OROOM
    for _ in range(n_rooms-2):
        build_room(lev, pool, 1, rng)
    # RANDOM_CORRIDORS
    makecorridors(lev, rng)
    # hero branch
    hero_room, hero = find_branch_room(lev, rng)
    return lev, hero_room, hero, rng
