const pdfjsLib = await import('../vendor/pdfjs/build/pdf.mjs');
globalThis.pdfjsLib = pdfjsLib;

const { EventBus, PDFLinkService, PDFViewer } = await import('../vendor/pdfjs/web/pdf_viewer.mjs');

pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
    '../vendor/pdfjs/build/pdf.worker.mjs',
    import.meta.url,
).href;

const params = new URLSearchParams(location.search);
const requestedFile = params.get('file') ?? '';
const teamName = params.get('name') ?? 'Technical report';
const sourceUrl = new URL(requestedFile, location.href);
const reportsRoot = new URL('../../reports/', import.meta.url);
const allowed = sourceUrl.origin === location.origin && sourceUrl.pathname.startsWith(reportsRoot.pathname);

const status = document.getElementById('viewer-status');
const viewerContainer = document.getElementById('viewerContainer');
const viewerElement = document.getElementById('viewer');
const pageNumber = document.getElementById('page-number');
const pageCount = document.getElementById('page-count');
const openLink = document.getElementById('open-pdf');
const downloadLink = document.getElementById('download-pdf');
document.getElementById('viewer-title').textContent = teamName;

if (!allowed) {
    status.textContent = 'This report link is invalid.';
    status.classList.add('error');
    throw new Error('Blocked invalid report URL.');
}

openLink.href = sourceUrl.href;
downloadLink.href = sourceUrl.href;
downloadLink.download = `${teamName.replace(/[^a-z0-9]+/gi, '-').replace(/^-|-$/g, '').toLowerCase()}-report.pdf`;

const eventBus = new EventBus();
const linkService = new PDFLinkService({ eventBus });
const staticL10n = {
    pause() {},
    resume() {},
};
const pdfViewer = new PDFViewer({
    container: viewerContainer,
    viewer: viewerElement,
    eventBus,
    linkService,
    l10n: staticL10n,
    textLayerMode: 1,
    removePageBorders: false,
});
linkService.setViewer(pdfViewer);

function fitWidth() {
    try {
        pdfViewer.currentScaleValue = 'page-width';
    } catch (error) {
        console.warn('The report could not be fitted to the reader width.', error);
    }
}

eventBus.on('pagesinit', () => {
    status.hidden = true;
    fitWidth();
});

eventBus.on('pagesloaded', ({ pagesCount }) => {
    pageCount.textContent = String(pagesCount);
});

eventBus.on('pagechanging', ({ pageNumber: currentPage }) => {
    pageNumber.value = String(currentPage);
});

pageNumber.addEventListener('change', () => {
    const requested = Number.parseInt(pageNumber.value, 10);
    if (Number.isFinite(requested)) {
        pdfViewer.currentPageNumber = Math.max(1, Math.min(requested, pdfViewer.pagesCount));
    }
});

document.getElementById('previous-page').addEventListener('click', () => {
    pdfViewer.currentPageNumber = Math.max(1, pdfViewer.currentPageNumber - 1);
});
document.getElementById('next-page').addEventListener('click', () => {
    pdfViewer.currentPageNumber = Math.min(pdfViewer.pagesCount, pdfViewer.currentPageNumber + 1);
});
document.getElementById('zoom-out').addEventListener('click', () => pdfViewer.decreaseScale());
document.getElementById('zoom-in').addEventListener('click', () => pdfViewer.increaseScale());
document.getElementById('fit-width').addEventListener('click', () => {
    fitWidth();
});

try {
    const loadingTask = pdfjsLib.getDocument({
        url: sourceUrl.href,
        cMapUrl: new URL('../vendor/pdfjs/cmaps/', import.meta.url).href,
        cMapPacked: true,
        standardFontDataUrl: new URL('../vendor/pdfjs/standard_fonts/', import.meta.url).href,
        wasmUrl: new URL('../vendor/pdfjs/wasm/', import.meta.url).href,
    });
    const pdfDocument = await loadingTask.promise;
    pdfViewer.setDocument(pdfDocument);
    linkService.setDocument(pdfDocument, null);
    pageCount.textContent = String(pdfDocument.numPages);
    await pdfViewer.pagesPromise;
    status.hidden = true;
    fitWidth();
} catch (error) {
    status.hidden = false;
    status.classList.add('error');
    status.innerHTML = 'The embedded reader could not open this report. <a href="' + sourceUrl.href + '" target="_blank" rel="noreferrer">Open the PDF directly</a>.';
    console.error(error);
}
