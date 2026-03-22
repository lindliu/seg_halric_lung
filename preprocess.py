import numpy as np
# import torch
import pydicom
import matplotlib.pylab as plt
import glob
import os 
import re
import tifffile as tiff


# print(torch.cuda.is_available())


def get_sorted_dicom_paths_3d(folder: str, series_uid: str | None = None):
    """
    Step-1: Scan folder, pick a SeriesInstanceUID (default: the one with most files),
            then return DICOM file paths sorted by slice order (z).

    Returns:
      sorted_paths: list[str]
      picked_series_uid: str
    """
    # collect candidate dicom files and their series uid
    items = []  # (series_uid, path)
    for fn in os.listdir(folder):
        p = os.path.join(folder, fn)
        if os.path.isdir(p):
            continue
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            uid = getattr(ds, "SeriesInstanceUID", None)
            if uid is not None:
                items.append((uid, p))
        except Exception:
            pass

    if not items:
        raise ValueError("No DICOM files found (or unreadable) in folder.")

    # choose series uid
    if series_uid is None:
        counts = {}
        for uid, _ in items:
            counts[uid] = counts.get(uid, 0) + 1
        series_uid = max(counts, key=counts.get)

    paths = [p for uid, p in items if uid == series_uid]
    if not paths:
        raise ValueError(f"No DICOM files found for SeriesInstanceUID={series_uid}")

    # sort by z (preferred), fallback slice location, fallback instance number
    def sort_key(path: str):
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            return float(ipp[2])
        sl = getattr(ds, "SliceLocation", None)
        if sl is not None:
            return float(sl)
        return float(getattr(ds, "InstanceNumber", 0))

    sorted_paths = sorted(paths, key=sort_key)

    # --- spacing/origin (best-effort) ---
    ds0 = pydicom.dcmread(paths[0], stop_before_pixels=True, force=True)
    px = getattr(ds0, "PixelSpacing", [1.0, 1.0])
    dy, dx = float(px[0]), float(px[1])
    dz = float(getattr(ds0, "SliceThickness", 1.0))

    spacing_zyx = (dz, dy, dx)

    return sorted_paths, series_uid, spacing_zyx

def dcm_to_tif_8bit(dcm_path, tif_path, step_y=1, step_x=1, crop=None, apply_rescale=False):
    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)

    # optional rescale
    if apply_rescale:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

    # crop = [650,1650,400,1400]
    if crop is not None:
        arr = arr[crop[0]:crop[1],crop[2]:crop[3]] # [325:825,200:700] #
    
    arr = arr[::step_y, ::step_x]

    # percentile window for display
    lo, hi = np.percentile(arr, (.1, 99.9))
    # lo, hi = arr.min(), arr.max()
    arr = np.clip(arr, lo, hi)
    arr8 = ((arr - lo) / (hi - lo + 1e-10) * 255.0).astype(np.uint8)

    tiff.imwrite(tif_path, arr8)
    

def dcm_to_tif_float32(dcm_path, tif_path, step_y=1, step_x=1, apply_rescale=False):
    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)

    # Keep physical/intended values (if present)
    if apply_rescale:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

    # arr = arr[650:1650,400:1400]
    arr = arr[::step_y, ::step_x]

    # Save exactly (no normalization, no 65535 scaling)
    tiff.imwrite(tif_path, arr.astype(np.float32))


# specify your image path
# root_path = glob.glob('./data/Rat MIR/Rat 19_during_VILI')[0]
# root_path = glob.glob('./data/Rat MIR/Rat 19_post_VILI')[0]
# root_path = glob.glob('./data/Rat MIR/Rat 18')[0]
# root_paths = glob.glob('./data/Rat MIR/Rat 11_baseline_11')
root_paths = [
            # './data/Rat MIR/Rat 9_during-VILI_9',
            # './data/Rat MIR/Rat 9_post_VILI_9',
            # './data/Rat MIR/Rat 11_during-VILI_11',
            # './data/Rat MIR/Rat 11_post-VILI_11',
            # './data/Rat MIR/Rat 14_during-VILI_14',
            # './data/Rat MIR/Rat 14_post-VILI_14',
            # './data/Rat MIR/Rat 16_during-VILI_16',
            './data/Rat MIR/Rat 19_during-VILI_19',
            './data/Rat MIR/Rat 19_post-VILI_19']
# root_paths = ['./data/Rat MIR/Rat 4', './data/Rat MIR/Rat 20']
# root_paths = ['./data/Rat MIR/Rat 12']
# root_paths = ['./data/Rat MIR/Rat 13']

# root_paths = glob.glob('./data/Rat MIR/*')
print(root_paths)
for root_path in root_paths:
    print(root_path)

    data_name = os.path.split(root_path)[1]


    folder = '1_original'
    sorted_paths, series_uid, spacing_zyx = get_sorted_dicom_paths_3d(os.path.join(root_path,folder))

    print('spacing_zyx: ', spacing_zyx)


    tif_path = os.path.join(root_path, '2_tif')
    os.makedirs(tif_path, exist_ok=True)

    assert int(0.1572/spacing_zyx[0])==0.1572/spacing_zyx[0]
    assert int(0.1572/spacing_zyx[1])==0.1572/spacing_zyx[1]
    assert int(0.1572/spacing_zyx[2])==0.1572/spacing_zyx[2]

    step_z = int(0.1572/spacing_zyx[0])
    step_y = int(0.1572/spacing_zyx[1])
    step_x = int(0.1572/spacing_zyx[2])

    # crop = None
    crop = [670,670+1024, 390,390+1024]
    # crop = [620,620+900, 390,390+1024] # 9,11,14,16 during and post
    # crop = [800,800+800, 600,600+860]  # 4, 20
    # crop = [800,800+800, 620,620+900]  # 12
    # crop = [600,600+800, 430,430+900]  # 13
    step_z,step_y,step_x = 1,1,1

    sorted_paths = sorted_paths[::step_z]
    for i in range(len(sorted_paths)):
        dcm_path = sorted_paths[i]
        
        tif_path_ = os.path.join(tif_path, data_name+f'_tif_{i:04d}.tif')
        
        # dcm_to_tif_float32(dcm_path_, tif_path_)
        dcm_to_tif_8bit(dcm_path, tif_path_, step_y=step_y, step_x=step_x, crop=crop)



    # # plt.imshow(ds.pixel_array)


