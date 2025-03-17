import cv2
import tensorflow as tf

def load_video(path: str, target_frames: int = 75, target_size: tuple = (46, 140)) -> tf.Tensor:
    """
    Loads and preprocesses a video.
    """
    cap = cv2.VideoCapture(path)
    frames = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cropped_frame = frame[190:236, 80:220] 
            resized_frame = cv2.resize(cropped_frame, target_size, interpolation=cv2.INTER_AREA)
            frames.append(resized_frame)

        cap.release()

        # Convert to TensorFlow tensor
        frames = tf.convert_to_tensor(frames, dtype=tf.float32)

        # Adjust frame count
        if len(frames) < target_frames:
            padding = tf.zeros((target_frames - len(frames), *target_size), dtype=tf.float32)
            frames = tf.concat([frames, padding], axis=0)
        elif len(frames) > target_frames:
            frames = frames[:target_frames]

        # Normalize
        mean = tf.math.reduce_mean(frames)
        std = tf.math.reduce_std(frames)
        frames = (frames - mean) / (std + 1e-8)

        # Add channel dimension
        frames = tf.expand_dims(frames, axis=-1)

        return frames

    except Exception as e:
        cap.release()
        print(f"Error processing video: {e}")
        return None
    
def predict_lip_pattern(video_path: str, model, num_to_char):
    try:
        video_frames = load_video(video_path)
        if video_frames is None:
            print("Failed to process video.")
            return None

        video_frames = tf.convert_to_tensor(video_frames, dtype=tf.float32)
        input_frames = tf.expand_dims(video_frames, axis=0)  # Add batch dimension

        # Predict using the trained model
        yhat = model.predict(input_frames)
        sequence_length = [yhat.shape[1]]

        # Decode predictions
        decoded = tf.keras.backend.ctc_decode(yhat, input_length=sequence_length, greedy=True)[0][0].numpy()
        predicted_text = [
            tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded
        ]

        return predicted_text[0].numpy().decode("utf-8")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

