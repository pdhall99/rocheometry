function doctestToPython(transcript) {
    const output = [];

    for (const line of transcript.replace(/\r\n?/g, "\n").split("\n")) {
        if (line === ">>>" || line.startsWith(">>> ")) {
            output.push(line === ">>>" ? "" : line.slice(4));
        } else if (line === "..." || line.startsWith("... ")) {
            output.push(line === "..." ? "" : line.slice(4));
        }

        // Ignore expected output.
    }

    return output.join("\n").trimEnd() + "\n";
}

document$.subscribe(() => {
    for (const block of document.querySelectorAll(".highlight.language-pycon")) {
        const code = block.querySelector("pre > code");
        if (!code) continue;

        block.dataset.copy = doctestToPython(code.textContent);
    }
});
