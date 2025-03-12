import DefaultTheme from "vitepress/theme";
import "./custom.css";

import TLMViewerAsync from "../../components/TLMViewerAsync.vue";
import TLMViewer from "../../components/TLMViewer.vue";
import LogoTitle from "../../components/LogoTitle.vue";
import Badges from "../../components/Badges.vue";

// Overwrite default layout component
import CustomLayout from "../../components/CustomLayout.vue";

/** @type {import('vitepress').Theme} */
export default {
  extends: DefaultTheme,
  Layout: CustomLayout,
  enhanceApp({ app }) {
    // register your custom global components
    app.component("TLMViewerAsync", TLMViewerAsync);
    app.component("TLMViewer", TLMViewer);
    app.component("LogoTitle", LogoTitle);
    app.component("Badges", Badges);
  },
};
