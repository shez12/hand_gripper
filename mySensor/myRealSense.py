import cv2
import numpy as np
import pyrealsense2 as rs
import datetime

def get_timestamp():
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    return timestamp

def depth_to_colormap(depth_image):
    return cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

class MyRealSense:
    def __init__(self):
        # Initialize the RealSense pipeline and configure streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Configure RealSense streams
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Start the pipeline
        self.pipeline.start(self.config)
        
        # Align depth to color
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

    def get_frames(self):
        """
        Waits for frames from the RealSense camera and aligns the depth frame to the color frame.
        Returns the color and depth images.
        """
        frames_ = self.pipeline.wait_for_frames()
        frames = self.align.process(frames_)

        # Get depth and color frames
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        # Convert frames to numpy arrays
        self.depth_image = np.asanyarray(depth_frame.get_data())
        self.color_image = np.asanyarray(color_frame.get_data())

        return self.color_image, self.depth_image

    def stop(self):
        """ Stop the RealSense pipeline """
        self.pipeline.stop()

    def concat_view(self):
        self.depth_colormap = self.depth_color_frame
        return  cv2.hconcat([self.color_image, self.depth_colormap])
        # return combined_image



# Initialize video writers for saving RGB and depth videos
def initialize_video_writers(timestamp):
    video_rgb_save_path = f'data/{timestamp}_rgb.mp4'
    video_depth_save_path = f'data/{timestamp}_depth.mp4'
    
    fourcc_rgb = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc_depth = cv2.VideoWriter_fourcc(*'mp4v')
    
    out_rgb = cv2.VideoWriter(video_rgb_save_path, fourcc_rgb, 30.0, (640, 480))
    out_depth = cv2.VideoWriter(video_depth_save_path, fourcc_depth, 30.0, (640, 480))
    
    return out_rgb, out_depth

# Main function to run the pipeline
if __name__ == "__main__":
    timestamp = get_timestamp()
    # Initialize the RealSense wrapper (MyRS class)
    my_rs = MyRealSense()

    # Initialize video writers
    out_rgb, out_depth = initialize_video_writers(timestamp)

    # Initialize video control variables
    recording = False

    try:
        while True:
            # Get color and depth frames from the RealSense camera
            color_image, depth_image = my_rs.get_frames()
            if color_image is None or depth_image is None:
                continue  # Skip if no frames are available
            depth_colormap = depth_to_colormap(depth_image)

            # Check for key press events
            key = cv2.waitKey(1) & 0xFF
            # Start recording video with blank key
            if key == ord(' '):
                if not recording:
                    recording = True
                    print("Recording started (via keyboard input)")
                else: 
                    recording = False
                    print("Recording ended (via keyboard input)")
                    break

            # If video recording has started, save frames to video files
            if recording:
                cv2.circle(color_image, (30, 30), 15, (0, 0, 255), -1)
                out_rgb.write(color_image)
                out_depth.write(depth_colormap)

            cv2.imshow("view", cv2.hconcat([color_image, depth_colormap]))


    finally:
        # Stop the RealSense pipeline and release resources
        my_rs.stop()
        cv2.destroyAllWindows()
        out_rgb.release()
        out_depth.release()



