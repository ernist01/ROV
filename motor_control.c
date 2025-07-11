#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <string.h>
#include <mavlink.h>

// Serial port for communication with Pixhawk (USB or UART)
#define SERIAL_PORT "/dev/ttyACM0"  // Change to your serial port
#define BAUD_RATE B57600  // Common baud rate for Pixhawk

// Set up serial port for communication with Pixhawk
int setup_serial(const char *port, int baud_rate) {
    int serial_fd = open(port, O_RDWR | O_NOCTTY | O_NDELAY);
    if (serial_fd == -1) {
        perror("Unable to open serial port");
        return -1;
    }

    struct termios options;
    tcgetattr(serial_fd, &options);
    cfsetispeed(&options, baud_rate);
    cfsetospeed(&options, baud_rate);

    options.c_cflag |= (CLOCAL | CREAD);
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;  // 8 data bits
    options.c_cflag &= ~CSTOPB; // 1 stop bit
    options.c_cflag &= ~PARENB; // No parity

    tcsetattr(serial_fd, TCSANOW, &options);
    return serial_fd;
}

// Send MAVLink message to Pixhawk
void send_mavlink_message(int serial_fd, mavlink_message_t *msg) {
    uint8_t buf[MAVLINK_MAX_PACKET_LEN];
    uint16_t len = mavlink_msg_to_send_buffer(buf, msg);
    write(serial_fd, buf, len);
}

// Set motor PWM using SET_SERVO message
void set_motor_pwm(int serial_fd, uint8_t channel, uint16_t pwm_value) {
    mavlink_message_t msg;
    mavlink_msg_set_servo_pack(1, 200, &msg, channel, pwm_value);  // Channel 1-8, PWM value 1000-2000
    send_mavlink_message(serial_fd, &msg);
}

int main() {
    // Open the serial port to communicate with Pixhawk
    int serial_fd = setup_serial(SERIAL_PORT, BAUD_RATE);
    if (serial_fd == -1) {
        return -1;
    }

    // Set motor 1 (channel 1) to PWM value of 1500 (neutral position)
    printf("Setting motor 1 to PWM 1500\n");
    set_motor_pwm(serial_fd, 1, 1500);
    usleep(1000000); // Wait for 1 second

    // Set motor 1 to PWM value of 2000 (full speed forward)
    printf("Setting motor 1 to PWM 2000\n");
    set_motor_pwm(serial_fd, 1, 2000);
    usleep(1000000); // Wait for 1 second

    // Set motor 1 to PWM value of 1000 (full speed backward)
    printf("Setting motor 1 to PWM 1000\n");
    set_motor_pwm(serial_fd, 1, 1000);
    usleep(1000000); // Wait for 1 second

    // Close the serial port
    close(serial_fd);
    return 0;
}
