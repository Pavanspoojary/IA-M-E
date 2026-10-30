"""File loader module for bluebook documents.

Supports JPG, JPEG, PNG images and scanned PDF documents.
Handles byte streams and disk files for batch processing.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import io
import cv2
import numpy as np
from PIL import Image

from src.pdf_processor import PDFProcessor, PDFProcessingError


@dataclass
class LoadedDocument:
    """Represents a loaded examination document page ready for image processing."""

    filename: str
    file_type: str  # 'image' or 'pdf'
    page_number: int  # 1-indexed
    total_pages: int
    image_bgr: Optional[np.ndarray]
    status: str  # 'ready', 'failed'
    error_message: Optional[str] = None


class FileLoader:
    """Loads and standardizes bluebook document files into OpenCV images."""

    SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
    SUPPORTED_PDF_EXTS = {".pdf"}

    def __init__(self, pdf_processor: Optional[PDFProcessor] = None):
        """Initialize FileLoader.

        Args:
            pdf_processor: PDF rendering instance.
        """
        self.pdf_processor = pdf_processor or PDFProcessor()

    def load_from_bytes(
        self,
        filename: str,
        file_bytes: bytes,
        target_page: int = 2,
    ) -> List[LoadedDocument]:
        """Load document(s) from memory bytes.

        Args:
            filename: Original name of the uploaded file.
            file_bytes: Raw file content in bytes.
            target_page: 1-indexed target page for PDFs (e.g., 2 for second sheet).

        Returns:
            List of LoadedDocument items.
        """
        ext = Path(filename).suffix.lower()

        if ext in self.SUPPORTED_IMAGE_EXTS:
            return [self._load_single_image(filename, file_bytes)]
        elif ext in self.SUPPORTED_PDF_EXTS:
            return self._load_pdf(filename, file_bytes, target_page=target_page)
        else:
            return [
                LoadedDocument(
                    filename=filename,
                    file_type="unknown",
                    page_number=1,
                    total_pages=0,
                    image_bgr=None,
                    status="failed",
                    error_message=f"Unsupported format '{ext}'. Allowed: JPG, JPEG, PNG, PDF.",
                )
            ]

    def load_from_path(
        self,
        file_path: str | Path,
        target_page: int = 2,
    ) -> List[LoadedDocument]:
        """Load document(s) from a local filesystem path.

        Args:
            file_path: Path to the target file.
            target_page: Target page for PDFs.

        Returns:
            List of LoadedDocument items.
        """
        p = Path(file_path)
        if not p.exists():
            return [
                LoadedDocument(
                    filename=p.name,
                    file_type="unknown",
                    page_number=1,
                    total_pages=0,
                    image_bgr=None,
                    status="failed",
                    error_message=f"File not found: {file_path}",
                )
            ]

        with open(p, "rb") as f:
            data = f.read()
        return self.load_from_bytes(p.name, data, target_page=target_page)

    def _load_single_image(self, filename: str, file_bytes: bytes) -> LoadedDocument:
        """Decode image bytes into OpenCV BGR numpy array."""
        try:
            # First attempt cv2.imdecode
            arr = np.frombuffer(file_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if img is None:
                # Fallback to Pillow in case of unusual encoding/color profile
                pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            return LoadedDocument(
                filename=filename,
                file_type="image",
                page_number=1,
                total_pages=1,
                image_bgr=img,
                status="ready",
                error_message=None,
            )
        except Exception as exc:
            return LoadedDocument(
                filename=filename,
                file_type="image",
                page_number=1,
                total_pages=1,
                image_bgr=None,
                status="failed",
                error_message=f"Failed to decode image: {exc}",
            )

    def _load_pdf(
        self,
        filename: str,
        file_bytes: bytes,
        target_page: int = 2,
    ) -> List[LoadedDocument]:
        """Render target page from PDF bytes."""
        try:
            total_pages = self.pdf_processor.get_page_count(file_bytes)
            # If target page exceeds total pages, attempt page 1 or log error
            selected_page = target_page
            if selected_page > total_pages:
                if total_pages == 1:
                    selected_page = 1
                else:
                    return [
                        LoadedDocument(
                            filename=filename,
                            file_type="pdf",
                            page_number=target_page,
                            total_pages=total_pages,
                            image_bgr=None,
                            status="failed",
                            error_message=(
                                f"Requested sheet {target_page}, but document only has "
                                f"{total_pages} page(s)."
                            ),
                        )
                    ]

            img_bgr = self.pdf_processor.render_page(file_bytes, page_number=selected_page)
            return [
                LoadedDocument(
                    filename=filename,
                    file_type="pdf",
                    page_number=selected_page,
                    total_pages=total_pages,
                    image_bgr=img_bgr,
                    status="ready",
                    error_message=None,
                )
            ]
        except PDFProcessingError as p_err:
            return [
                LoadedDocument(
                    filename=filename,
                    file_type="pdf",
                    page_number=target_page,
                    total_pages=0,
                    image_bgr=None,
                    status="failed",
                    error_message=str(p_err),
                )
            ]
        except Exception as exc:
            return [
                LoadedDocument(
                    filename=filename,
                    file_type="pdf",
                    page_number=target_page,
                    total_pages=0,
                    image_bgr=None,
                    status="failed",
                    error_message=f"Unexpected error reading PDF: {exc}",
                )
            ]
