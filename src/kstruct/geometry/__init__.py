from OCP.Message import Message, Message_Gravity

from kstruct.geometry.step_import import GeometryImportError, load_cad
from kstruct.geometry.tessellate import faces_of, tessellate

# OpenCASCADE prints STEP transfer statistics to stdout; keep only real failures
for _printer in Message.DefaultMessenger_s().Printers():
    _printer.SetTraceLevel(Message_Gravity.Message_Fail)

__all__ = ["GeometryImportError", "faces_of", "load_cad", "tessellate"]
