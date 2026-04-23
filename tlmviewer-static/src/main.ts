import tlmviewer from "tlmviewer";

const overlay = document.getElementById("overlay") as HTMLElement;
const viewer = document.getElementById("viewer") as HTMLElement;
const errorBanner = document.getElementById("error-banner") as HTMLElement;
const fileInput = document.getElementById("file-input") as HTMLInputElement;
const browseBtn = document.getElementById("browse-btn") as HTMLButtonElement;
const loadAnotherBtn = document.getElementById("load-another-btn") as HTMLButtonElement;

function showError(msg: string) {
    errorBanner.textContent = msg;
    errorBanner.style.display = "block";
}

function hideError() {
    errorBanner.style.display = "none";
}

async function loadFile(file: File) {
    hideError();
    try {
        const text = await file.text();
        const data = JSON.parse(text);
        data.controls = { ...data.controls, show_controls: true };
        tlmviewer.embed(viewer, JSON.stringify(data));
        overlay.classList.add("loaded");
    } catch (e) {
        showError(`Error loading scene: ${e}`);
        overlay.classList.remove("loaded");
    }
}

browseBtn.addEventListener("click", () => fileInput.click());
loadAnotherBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) loadFile(file);
    fileInput.value = "";
});

document.addEventListener("dragover", (e) => {
    e.preventDefault();
    overlay.classList.add("drag-active");
});

document.addEventListener("dragleave", (e) => {
    if (!e.relatedTarget) overlay.classList.remove("drag-active");
});

document.addEventListener("drop", (e) => {
    e.preventDefault();
    overlay.classList.remove("drag-active");
    const file = e.dataTransfer?.files[0];
    if (file) loadFile(file);
});
