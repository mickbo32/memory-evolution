import imageio as iio
import os


def generate_gif_from_path(frames_dir,
                           gif_name="frames.gif",
                           remove_frames: bool = False,
                           save_gif_in_frames_dir: bool = True,
                           **kwargs):
    """Get all the frames in ``frames_dir`` and generate a gif.
    The gif is saved in the same path of the frames.

    If ``remove_frames`` is True, it will delete all frames
    once the gif file have been generated. If `frames_dir``
    is empty after deleting frames, also that will be deleted.

    If ``save_gif_in_frames_dir`` is True, save the gif in the
    same directory where frames are; if False, use ``gif_name`` as
    an independent path relative to the working directory or a
    full path to the gif file (if ``save_gif_in_frames_dir`` is False
    and ``gif_name`` it is just a name, the gif will be saved
    in the current working directory).

    **kwargs are passed to iio.get_writer(), e.g. duration=0.04 or fps=25
    """
    print(f"frames_dir: {frames_dir}")
    print(f"gif_name: {gif_name}")
    if not gif_name.endswith('.gif'):
        raise ValueError("gif name should ends with '.gif'")
    # frames_filenames = sorted(os.listdir(frames_dir))  # use os.walk() if you want to go inside subdirectories.
    frames_paths = [os.path.join(frames_dir, f"frame_{i}.jpg") for i in range(len(os.listdir(frames_dir)))]
    if save_gif_in_frames_dir:
        gif_name = os.path.join(frames_dir, gif_name)
    print('Generating the gif image...')
    with iio.get_writer(gif_name, mode="I", **kwargs) as writer:
        for idx, path in enumerate(frames_paths):
            frame = iio.imread(path)
            writer.append_data(frame)
            if remove_frames:
                os.remove(path)
    if remove_frames and not os.listdir(frames_dir):
        # if 'frames_dir' is empty
        os.rmdir(frames_dir)
    print('Gif image generated.\n')

