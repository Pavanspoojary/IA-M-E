"""PDF processing module using PyMuPDF.

Renders scanned examination bluebook PDF documents into OpenCV images.
"""

from typing import List, Tuple, Optional
import numpy as np
import pymupdf
import cv2


class PDFProcessingError(Exception):
    """Exception raised when PDF processing fails."""
    pass


class PDFProcessor:
    """Renders PDF pages to OpenCV BGR images at consistent resolutions."""

    def __init__(self, default_dpi: int = 150):
        """Initialize PDFProcessor.

        Args:
            default_dpi: Rendering resolution in dots per inch (default 150).
        """
        self.default_dpi = default_dpi

    def get_page_count(self, pdf_bytes_or_path: bytes | str) -> int:
        """Get the total number of pages in the PDF document.

        Args:
            pdf_bytes_or_path: File bytes or string path to PDF file.

        Returns:
            Number of pages in the PDF.
        """
        doc = None
        try:
            if isinstance(pdf_bytes_or_path, bytes):
                doc = pymupdf.open(stream=pdf_bytes_or_path, filetype="pdf")
            else:
                doc = pymupdf.open(pdf_bytes_or_path)
            return len(doc)
        except Exception as exc:
            raise PDFProcessingError(f"Failed to open PDF document: {exc}") from exc
        finally:
            if doc is not None:
                doc.close()

    def render_page(
        self,
        pdf_bytes_or_path: bytes | str,
        page_number: int = 2,
        dpi: Optional[int] = None,
    ) -> np.ndarray:
        """Render a specific 1-indexed page of a PDF as an OpenCV BGR image.

        Args:
            pdf_bytes_or_path: File bytes or string path to PDF file.
            page_number: 1-indexed page number to render (default 2 for second sheet).
            dpi: Resolution in DPI. Uses self.default_dpi if omitted.

        Returns:
            OpenCV image as a NumPy uint8 BGR array.
        """
        doc = None
        target_dpi = dpi or self.default_dpi
        zoom = target_dpi / 72.0  # Standard PDF point resolution is 72 dpi
        matrix = pymupdf.Matrix(zoom, zoom)

        try:
            if isinstance(pdf_bytes_or_path, bytes):
                doc = pymupdf.open(stream=pdf_bytes_or_path, filetype="pdf")
            else:
                doc = pymupdf.open(pdf_bytes_or_path)

            total_pages = len(doc)
            if total_pages == 0:
                raise PDFProcessingError("The provided PDF has 0 pages.")

            # Validate page number bounds (1-indexed)
            if page_number < 1 or page_number > total_pages:
                raise PDFProcessingError(
                    f"Page {page_number} requested, but PDF contains {total_pages} pages."
                )

            page = doc[page_number - 1]
            pix = page.get_pixmap(matrix=matrix, alpha=False)

            # Convert pixmap bytes to numpy BGR image for OpenCV
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 3:  # RGB
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            elif pix.n == 1:  # Grayscale
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            elif pix.n == 4:  # RGBA
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            else:
                raise PDFProcessingError(f"Unsupported pixmap channel count: {pix.n}")

            return img_bgr

        except PDFProcessingError:
            raise
        except Exception as exc:
            raise PDFProcessingError(f"Error rendering PDF page {page_number}: {exc}") from exc
        finally:
            if doc is not None:
                doc.close()

    def render_all_pages(
        self,
        pdf_bytes_or_path: bytes | str,
        dpi: Optional[int] = None,
    ) -> List[Tuple[int, np.ndarray]]:
        """Render all pages of a PDF document.

        Args:
            pdf_bytes_or_path: File bytes or string path to PDF file.
            dpi: Resolution in DPI.

        Returns:
            List of tuples (page_number_1_indexed, img_bgr).
        """
        total = self.get_page_count(pdf_bytes_or_path)
        pages: List[Tuple[int, np.ndarray]] = []
        for p in range(1, total + 1):
            img = self.render_page(pdf_bytes_or_path, page_number=p, dpi=dpi)
            pages.append((p, img))
        return pages
