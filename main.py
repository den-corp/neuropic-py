from concurrent import futures
import logging

import sys 
sys.path.append("./proto")

import grpc
import proto.neuroservice_pb2 as proto
import proto.neuroservice_pb2_grpc as proto_grpc


class NeuroService(proto_grpc.NeuroService):
    def GeneratePicture(self, request, context):
        #hello = b'\x48\x65\x6c\x6c\x6f'
        originalImage = request.OriginalImage
        return proto.GeneratePictureResponse(ResultImage=originalImage)


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