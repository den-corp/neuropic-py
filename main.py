from concurrent import futures
import logging

import sys 
sys.path.append("./proto")

import grpc
import proto.neuroservice_pb2 as proto
import proto.neuroservice_pb2_grpc as proto_grpc

import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import io
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import cv2

from generate import NeuralStyleTransfer

def save_bytes_as_image(image_bytes, file_path):
    image_stream = BytesIO(image_bytes)
    image = Image.open(image_stream)
    image.save(file_path)

def start_neural_style_transfer(content_image_path, style_image_path, iterations):
    neural_style_transfer = NeuralStyleTransfer(content_image_path, style_image_path)
    result = neural_style_transfer.style_transfer(iterations, 1e3, 1e-2)

    return result

class NeuroService(proto_grpc.NeuroService):
    def GeneratePicture(self, request, context):
        #hello = b'\x48\x65\x6c\x6c\x6f'
        originalPath = "originalImage.jpg"
        originalImage = request.OriginalImage
        save_bytes_as_image(originalImage, originalPath)

        stylePath = "styleImage.jpg"
        styleImage = request.StyleImage
        save_bytes_as_image(styleImage, stylePath)

        iterations = 1
        result = start_neural_style_transfer(originalPath, stylePath, iterations)
        return proto.GeneratePictureResponse(ResultImage=result)


def serve():
    print("Server starting")
    port = "2000"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    proto_grpc.add_NeuroServiceServicer_to_server(NeuroService(), server)
    server.add_insecure_port("0.0.0.0:2000")
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()