import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
import serial
import struct
import time

class RosImageUART(Node):
    def __init__(self, port='/dev/ttyUSB0', baud=115200):
        super().__init__('ros_image_uart')
        self.subscription = self.create_subscription(Image, 'image_raw', self.image_callback, 10)
        self.subscription  # prevent unused variable warning
        self.dims = (64,64)  # resize dimensions

        # UART setup
        try:
            self.ser = serial.Serial(port, baud, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE)
            self.get_logger().info(f'UART opened on {port} at {baud} baud')
        except Exception as e:
            self.get_logger().error(f'Failed to open serial port {port}: {e}')
            self.ser = None

        # FPS tracking
        self.last_frame_time = None

    def image_callback(self, msg: Image):
        if self.ser is None:
            return

        # Input FPS
        now = time.time()
        if self.last_frame_time is None:
            input_fps = 0.0
        else:
            input_fps = 1.0 / (now - self.last_frame_time)
        self.last_frame_time = now

        encoding = msg.encoding.lower()
        try:
            if 'mono' in encoding:
                gray = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
            elif 'bgr8' in encoding or 'rgb8' in encoding:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif 'yuy' in encoding:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 2))
                gray = cv2.cvtColor(img, cv2.COLOR_YUV2GRAY_YUY2)
            else:
                self.get_logger().error(f'Unsupported encoding: {msg.encoding}')
                return
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        # Resize and normalize
        img_resized = cv2.resize(gray, self.dims, interpolation=cv2.INTER_AREA)
        img_norm = (img_resized / 255).astype('float32')

        # Send over UART
        start_process = time.time()
        for i in range(self.dims[1]):
            for j in range(self.dims[0]):
                self.ser.write(bytearray(struct.pack("f", img_norm[i][j])))

        # Wait for NN output
        nn_res = ""
        while "output" not in nn_res:
            try:
                nn_res = self.ser.readline().decode('utf-8').strip()
            except Exception:
                continue

        # Processing FPS
        process_time = time.time() - start_process
        process_fps = 1.0 / process_time if process_time > 0 else 0.0

        # Map numeric output to direction
        mapping = {'0': 'straight', '1': 'left', '2': 'right'}
        nn_num = nn_res[-1]
        direction = mapping.get(nn_num, f'Unknown ({nn_res})')

        # Print live status
        print(f'\rNN response: {direction} | Input FPS: {input_fps:.2f} | Processing FPS: {process_fps:.2f}', end='')

def main(args=None):
    rclpy.init(args=args)
    node = RosImageUART('/dev/ttyUSB1')  # replace with your serial port
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
