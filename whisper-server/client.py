import zmq

from loguru import logger


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    logger.debug("Connecting to hello world server...")

    socket.send_string("start")
    message = socket.recv().decode()
    print(f"Received reply: [ {message} ]")


if __name__ == "__main__":
    main()
