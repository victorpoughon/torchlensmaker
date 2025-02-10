async function importtlm() {
    try {
        return await import("/tlmviewer.js");
    } catch (error) {
        console.log("error", error);
        return await import("/files/test_notebooks/tlmviewer.js");
    }
}

const module = await importtlm();
const tlmviewer = module.tlmviewer;

const data = '$data';

setTimeout(() => {
    tlmviewer.embed(document.getElementById("$div_id"), data);    
}, 0);
