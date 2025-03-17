// Load the tlmviewer js library in a jupyter notebook context
async function importtlm() {
    const sources = [
        "/tlmviewer-$version.js",
        "/files/tlmviewer-${version}.js",
        "https://unpkg.com/tlmviewer@${version}",
    ];

    // Helper function to attempt loading from a source
    const tryImport = async (url) => {
        try {
            const module = await import(url);
            return module;
        } catch (error) {
            return null;
        }
    };

    // Try sources in sequence
    for (const url of sources) {
        const result = await tryImport(url);
        if (result){
            console.log("Successfully loaded tlmviewer from " + url);
            return result;
        }
    }

    // If all sources fail
    throw new Error('Failed to load tlmviewer');
}

const module = await importtlm();
const tlmviewer = module.default;

const data = '$data';

setTimeout(() => {
    tlmviewer.embed(document.getElementById("$div_id"), data);    
}, 0);
