from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class GeneratePictureRequest(_message.Message):
    __slots__ = ["OriginalImage", "StyleImage"]
    ORIGINALIMAGE_FIELD_NUMBER: _ClassVar[int]
    STYLEIMAGE_FIELD_NUMBER: _ClassVar[int]
    OriginalImage: bytes
    StyleImage: bytes
    def __init__(self, OriginalImage: _Optional[bytes] = ..., StyleImage: _Optional[bytes] = ...) -> None: ...

class GeneratePictureResponse(_message.Message):
    __slots__ = ["ResultImage"]
    RESULTIMAGE_FIELD_NUMBER: _ClassVar[int]
    ResultImage: bytes
    def __init__(self, ResultImage: _Optional[bytes] = ...) -> None: ...
