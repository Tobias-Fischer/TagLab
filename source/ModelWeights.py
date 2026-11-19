# TagLab
# Network weights are not part of the repository: they are downloaded from the TagLab web site the first time
# a tool needs them (the same files that install.py downloads).

import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox, QProgressDialog

from model_download import ModelDownloadCancelled, ModelDownloadError, ensure_model, looks_complete
from source.genutils import modelPath

MODELS_URL = "https://taglab.isti.cnr.it/models/"

# the weights that can be downloaded and their approximate size in MB
MODEL_SIZES_MB = {
    "dextr_corals.pth": 196,
    "deeplab-resnet.pth.tar": 475,
    "ritm_corals.pth": 41,
    "pocillopora.net": 238,
    "porites.net": 238,
    "pocillopora_porite_montipora.net": 238,
    "sam_vit_h_4b8939.pth": 2560,
}

# weights already checked in this session (the check asks the server for the size)
_checked = set()


def ensureModel(name, parent=None):
    """
    Returns the path of the weights file `name` in the models folder. If it is missing, or incomplete (an interrupted
    download, e.g. by an earlier install: see model_download.looks_complete), asks the user and downloads it (with a
    progress dialog). Returns None if the file is not available (declined, cancelled or download failed); in that case
    the user has already been told, so the caller only has to give up.
    """

    path = modelPath(name)
    url = MODELS_URL + name
    if os.path.exists(path):
        # weights that are not on the server (e.g. trained by the user) are used as they are
        if path in _checked or name not in MODEL_SIZES_MB or looks_complete(path, url):
            _checked.add(path)
            return path
        text = ("The network weights " + name + " needed by this tool are incomplete (" +
                str(os.path.getsize(path) // (1 << 20)) + " MB of about " + str(MODEL_SIZES_MB[name]) +
                " MB; an earlier download was probably interrupted).\n\nDownload them again now from " +
                MODELS_URL + " ?")
    elif name not in MODEL_SIZES_MB:
        QMessageBox.warning(parent, "Model not found", "The model file " + name + " is not in the models folder.")
        return None
    else:
        text = ("The network weights " + name + " (about " + str(MODEL_SIZES_MB[name]) + " MB) are needed by this "
                "tool and are not installed.\n\nDownload them now from " + MODELS_URL + " ?")

    reply = QMessageBox.question(parent, "Download model", text, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
    if reply != QMessageBox.Yes:
        return None

    dialog = QProgressDialog("Downloading " + name + "...", "Cancel", 0, MODEL_SIZES_MB[name], parent)
    dialog.setWindowTitle("Downloading model")
    dialog.setWindowModality(Qt.WindowModal)
    dialog.setMinimumDuration(0)
    dialog.setValue(0)

    cancelled = []
    dialog.canceled.connect(lambda: cancelled.append(True))

    def progress(read, total):
        dialog.setMaximum(max(1, total // (1 << 20)))
        dialog.setValue(min(read // (1 << 20), dialog.maximum() - 1))
        QApplication.processEvents()
        return not cancelled

    error = None
    try:
        # checks the download (size, and the CRC of ZIP checkpoints) before it replaces anything
        ensure_model(url, path, progress)
    except ModelDownloadCancelled:
        error = ""
    except ModelDownloadError as exc:
        error = str(exc)
    dialog.canceled.disconnect()
    dialog.close()

    if error is not None:
        if error:
            QMessageBox.warning(parent, "Download failed",
                                "Could not download " + name + ":\n" + error + "\n\nCheck the internet connection, "
                                "or download\n" + url + "\nand copy it into the folder " + os.path.dirname(path))
        return None

    _checked.add(path)
    return path
