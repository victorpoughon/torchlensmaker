<style scoped>
div#logo-title {
    max-width: 35rem;
    padding: 2rem 0.5rem;
    box-sizing: border-box;
    border-radius: 20px;
    box-shadow: 8px 10px 12px 0px rgba(0, 0, 0, 0.6);
    background: black;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    position: relative;
    margin: 0 auto;
    margin-bottom: 40px;
}

#logo-title h1 {
    margin: 0;
    font-size: calc(2.5rem + 0.4vw);
    text-align: center;
    display: inline-block;

    font-weight: bolder;

    background: linear-gradient(-90deg,
            rgba(56, 109, 249, 0.9) 0%,
            rgba(18, 200, 230, 0.9) 29%,
            rgba(90, 248, 200, 0.8) 43%,
            rgba(164, 248, 159, 0.8) 57%,
            rgba(236, 200, 111, 0.9) 71%,
            rgba(255, 109, 56, 0.9) 100%);
    filter: saturate(80%);

    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;

    /* text-transform: uppercase; */
}

@media screen and (min-width: 1200px) {
    h1 {
        font-size: 35px;
        /* Maximum size */
    }
}
</style>

<template>
    <div id="logo-title">
        <h1>Torch Lens Maker</h1>
    </div>
</template>
