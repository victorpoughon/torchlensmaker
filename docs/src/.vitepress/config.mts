import { defineConfig } from "vitepress";

// Vite plugins
import version from "vite-plugin-package-version";

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Torch Lens Maker",
  description: "Differentiable geometric optics in PyTorch",
  head: [
    [
      "link",
      {
        rel: "icon",
        href: "/torchlensmaker/logos/tlmlogo_black130_margin.png",
      },
    ],
  ],
  base: "/torchlensmaker/",
  themeConfig: {
    logo: "/logos/tlmlogo_black150.png",
    notFound: {
      quote: "Sorry!",
    },

    nav: [
      { text: "Home", link: "/" },
      { text: "Documentation", link: "/doc/getting-started" },
      { text: "Examples", link: "/examples" },
      // { text: "Blog", link: "/blog" },
      { text: "About", link: "/about" },
      {
        text: "Community",
        items: [
          {
            text: "GitHub Discussions",
            link: "https://github.com/victorpoughon/torchlensmaker/discussions",
          },
          { text: "Funding", link: "/about#funding" },
          { text: "Mailing List", link: "/about#newsletter" },
        ],
      },
    ],

    sidebar: {
      "/doc/": [
        {
          text: "Introduction",
          items: [
            { text: "Getting Started", link: "/doc/getting-started" },
            { text: "Installation", link: "/doc/installation" },
            { text: "Features", link: "/doc/features" },
          ],
        },
        {
          text: "Modeling",
          items: [
            { text: "Light Sources", link: "/doc/modeling/light_sources" },
            { text: "Surfaces", link: "/doc/modeling/surfaces" },
            { text: "Sampling", link: "/doc/modeling/sampling" },
          ],
        },
        {
          text: "Advanced Topics",
          items: [
            {
              text: "Collision detection",
              link: "/doc/advanced/collision_detection",
            },
          ],
        },
      ],

      "/examples/": [
        {
          text: "Examples",
          items: [
            { text: "Landscape Lens", link: "/examples/landscape" },
            { text: "Simple Lenses", link: "/examples/simple_lenses" },
            {
              text: "Simple Optimization",
              link: "/examples/simple_optimization",
            },
            { text: "Cooke Triplet", link: "/examples/cooke_triplet" },
            { text: "Snell's Window", link: "/examples/snells_window" },
            { text: "Pink Floyd", link: "/examples/pink_floyd" },
            { text: "Rainbow", link: "/examples/rainbow" },
            {
              text: "Reflecting Telescope",
              link: "/examples/reflecting_telescope",
            },
            { text: "Triple Biconvex", link: "/examples/triple_biconvex" },
            {
              text: "Variable Lens Sequence",
              link: "/examples/variable_lens_sequence",
            },
            { text: "Test notebooks", link: "/test_notebooks" },
          ],
        },
      ],

      "/blog/": [
        {
          text: "Blog",
          items: [],
        },
      ],
    },

    socialLinks: [
      {
        icon: "github",
        link: "https://github.com/victorpoughon/torchlensmaker",
      },
    ],

    search: {
      provider: "local",
    },
  },

  cleanUrls: true,

  markdown: {
    math: true,
    defaultHighlightLang: "python",
    theme: {
      light: "github-light",
      dark: "github-dark",
    },
  },

  vue: {
    template: {
      transformAssetUrls: {
        video: ["src", "poster"],
        source: ["src"],
        img: ["src"],
        image: ["xlink:href", "href"],
        use: ["xlink:href", "href"],
        TLMViewer: ["src"],
      },
    },
  },

  vite: {
    build: {
      assetsInlineLimit: 0,
      sourcemap: false,
      commonjsOptions: {
        sourceMap: false,
      },
    },
    plugins: [version()],
    server: { host: true },
  },
});
