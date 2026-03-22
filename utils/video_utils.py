import cv2

def get_frames(video_path, max_frames=5):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []

    if total_frames == 0:
        return frames

    # 🔥 pick evenly spaced frames
    indices = [int(i * total_frames / max_frames) for i in range(max_frames)]

    current_frame = 0
    target_idx = 0

    while cap.isOpened() and target_idx < len(indices):
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame == indices[target_idx]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            target_idx += 1

        current_frame += 1

    cap.release()
    return frames