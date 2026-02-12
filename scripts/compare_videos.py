"""Compare original vs our MiniHack environment videos.

Extracts frames from each pair, creates side-by-side comparison images,
and reports differences.
"""
import os
import sys
import numpy as np

try:
    import imageio.v3 as iio
except ImportError:
    import imageio as iio

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


ORIGINALS_DIR = '/home/renos/nethax/videos/originals'
OURS_DIR = '/home/renos/nethax/videos/ours'
COMPARE_DIR = '/home/renos/nethax/videos/comparisons'

# All envs we expect to compare
ENVS = [
    'Corridor2', 'Corridor3', 'Corridor5',
    'Mazewalk',
    'ExploreMazeEasy', 'ExploreMazeEasyPremapped',
    'ExploreMazeHard', 'ExploreMazeHardPremapped',
    'LavaCrossing',
    'HideNSeek', 'HideNSeekBig', 'HideNSeekLava', 'HideNSeekMapped',
    'QuestEasy',
    'LockedDoor', 'LockedDoorFixed',
    'QuestHard',
    'ClosedDoor',
    'MementoEasy', 'MementoShort', 'MementoHard',
    'Soko1a', 'Soko1b', 'Soko2a', 'Soko2b',
    'Soko3a', 'Soko3b', 'Soko4a', 'Soko4b',
]


def load_first_frame(video_path):
    """Load the first frame from an mp4 video."""
    try:
        frames = iio.imread(video_path)
        if hasattr(frames, '__len__') and len(frames) > 0:
            return np.array(frames[0])
        return np.array(frames)
    except Exception as e:
        print(f"  Error loading {video_path}: {e}")
        return None


def create_comparison(orig_frame, ours_frame, env_name, output_path):
    """Create a side-by-side comparison image."""
    if not HAS_PIL:
        # Fallback: simple numpy concat
        # Resize to same height
        oh, ow = orig_frame.shape[:2]
        mh, mw = ours_frame.shape[:2]
        target_h = max(oh, mh)

        # Pad shorter one
        if oh < target_h:
            pad = np.zeros((target_h - oh, ow, 3), dtype=np.uint8)
            orig_frame = np.vstack([orig_frame, pad])
        if mh < target_h:
            pad = np.zeros((target_h - mh, mw, 3), dtype=np.uint8)
            ours_frame = np.vstack([ours_frame, pad])

        # Add separator
        sep = np.ones((target_h, 4, 3), dtype=np.uint8) * 255
        combined = np.hstack([orig_frame, sep, ours_frame])
        iio.imwrite(output_path, combined)
        return

    # PIL version with labels
    oh, ow = orig_frame.shape[:2]
    mh, mw = ours_frame.shape[:2]
    label_h = 30
    sep_w = 10
    target_h = max(oh, mh)
    total_w = ow + sep_w + mw
    total_h = target_h + label_h

    img = Image.new('RGB', (total_w, total_h), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Labels
    draw.text((ow // 2 - 30, 5), "Original", fill=(255, 255, 255))
    draw.text((ow + sep_w + mw // 2 - 15, 5), "Ours", fill=(255, 255, 255))

    # Paste frames
    orig_img = Image.fromarray(orig_frame)
    ours_img = Image.fromarray(ours_frame)
    img.paste(orig_img, (0, label_h))
    img.paste(ours_img, (ow + sep_w, label_h))

    # Separator
    draw.rectangle([ow, label_h, ow + sep_w, total_h], fill=(128, 128, 128))

    img.save(output_path)


def analyze_env(env_name):
    """Analyze one env pair and return a report dict."""
    orig_path = os.path.join(ORIGINALS_DIR, f'original_{env_name}.mp4')
    ours_path = os.path.join(OURS_DIR, f'ours_{env_name}.mp4')

    report = {'env': env_name, 'status': 'OK', 'notes': []}

    # Check existence
    if not os.path.exists(orig_path):
        report['status'] = 'MISSING_ORIGINAL'
        report['notes'].append(f'No original video found')
        return report
    if not os.path.exists(ours_path):
        report['status'] = 'MISSING_OURS'
        report['notes'].append(f'No our video found')
        return report

    # Load frames
    orig_frame = load_first_frame(orig_path)
    ours_frame = load_first_frame(ours_path)

    if orig_frame is None:
        report['status'] = 'ERROR'
        report['notes'].append('Failed to load original frame')
        return report
    if ours_frame is None:
        report['status'] = 'ERROR'
        report['notes'].append('Failed to load our frame')
        return report

    # Size comparison
    orig_shape = orig_frame.shape
    ours_shape = ours_frame.shape
    report['orig_size'] = f'{orig_shape[1]}x{orig_shape[0]}'
    report['ours_size'] = f'{ours_shape[1]}x{ours_shape[0]}'

    if orig_shape != ours_shape:
        report['notes'].append(f'Size differs: orig={orig_shape[1]}x{orig_shape[0]}, ours={ours_shape[1]}x{ours_shape[0]}')

    # Count frames in each
    try:
        orig_frames = iio.imread(orig_path)
        ours_frames = iio.imread(ours_path)
        orig_nframes = len(orig_frames) if hasattr(orig_frames, '__len__') else 1
        ours_nframes = len(ours_frames) if hasattr(ours_frames, '__len__') else 1
        report['orig_frames'] = orig_nframes
        report['ours_frames'] = ours_nframes
    except:
        pass

    # Visual content analysis: check if our frame has visible content
    ours_mean = np.mean(ours_frame)
    orig_mean = np.mean(orig_frame)
    if ours_mean < 5:
        report['notes'].append('Our frame appears mostly black/empty')
    if orig_mean < 5:
        report['notes'].append('Original frame appears mostly black/empty')

    # Check if our frame has map-like structure (non-zero pixels spread around)
    ours_nonzero = np.count_nonzero(ours_frame.sum(axis=-1))
    orig_nonzero = np.count_nonzero(orig_frame.sum(axis=-1))
    total_pixels = orig_frame.shape[0] * orig_frame.shape[1]
    ours_total = ours_frame.shape[0] * ours_frame.shape[1]

    orig_fill = orig_nonzero / total_pixels * 100
    ours_fill = ours_nonzero / ours_total * 100
    report['orig_fill_pct'] = f'{orig_fill:.1f}%'
    report['ours_fill_pct'] = f'{ours_fill:.1f}%'

    # Create side-by-side comparison
    compare_path = os.path.join(COMPARE_DIR, f'compare_{env_name}.png')
    create_comparison(orig_frame, ours_frame, env_name, compare_path)

    if report['notes']:
        report['status'] = 'DIFF'

    return report


def main():
    os.makedirs(COMPARE_DIR, exist_ok=True)

    if len(sys.argv) > 1:
        envs = [sys.argv[1]]
    else:
        envs = ENVS

    reports = []
    for env_name in envs:
        print(f"Comparing {env_name}...", flush=True)
        report = analyze_env(env_name)
        reports.append(report)

    # Print summary
    print("\n" + "=" * 80)
    print("COMPARISON REPORT")
    print("=" * 80)

    print(f"\n{'Env':<30} {'Status':<15} {'Orig Size':<15} {'Ours Size':<15} {'Orig Fill':<10} {'Ours Fill':<10}")
    print("-" * 95)

    for r in reports:
        print(f"{r['env']:<30} {r['status']:<15} {r.get('orig_size','N/A'):<15} {r.get('ours_size','N/A'):<15} {r.get('orig_fill_pct','N/A'):<10} {r.get('ours_fill_pct','N/A'):<10}")
        for note in r.get('notes', []):
            print(f"  -> {note}")

    # Summary counts
    ok = sum(1 for r in reports if r['status'] == 'OK')
    diff = sum(1 for r in reports if r['status'] == 'DIFF')
    missing = sum(1 for r in reports if 'MISSING' in r['status'])
    error = sum(1 for r in reports if r['status'] == 'ERROR')

    print(f"\nSummary: {ok} OK, {diff} DIFF, {missing} MISSING, {error} ERROR")
    print(f"Comparison images saved to: {COMPARE_DIR}")


if __name__ == '__main__':
    main()
