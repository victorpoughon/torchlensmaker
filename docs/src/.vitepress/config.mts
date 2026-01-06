import { defineConfig, createContentLoader, type SiteConfig } from "vitepress";
import path from "path";
import { writeFileSync } from "fs";
import { Feed } from "feed";

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
  buildEnd: buildRSS,
  themeConfig: {
    logo: "/logos/tlmlogo_black150.png",
    notFound: {
      quote: "Sorry!",
    },

    footer: {
      message:
        'Released under the <a href="https://github.com/victorpoughon/torchlensmaker/blob/main/LICENSE">GPL-3.0 license</a>.',
      copyright:
        'Copyright © 2024-present <a href="https://victorpoughon.fr">Victor Poughon</a>',
    },

    nav: [
      { text: "Home", link: "/" },
      { text: "Documentation", link: "/doc/getting-started" },
      { text: "Examples", link: "/examples" },
      { text: "Blog", link: "/blog/2026-01-05-announcing-nlnet-funding" },
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
          items: [
            {
              text: "Torch Lens Maker receives NLnet grant",
              link: "/blog/2026-01-05-announcing-nlnet-funding",
            },
          ],
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

async function buildRSS(config: SiteConfig) {
  const hostname: string = "https://victorpoughon.github.io/torchlensmaker";

  const feed = new Feed({
    title: "",
    description: "Torch Lens Maker Blog",
    id: hostname,
    link: hostname,
    language: "en",
    copyright: "Copyright (c) 2024-present, Victor Poughon",
  });

  // You might need to adjust this if your Markdown files
  // are located in a subfolder
  const posts = await createContentLoader("blog/*.md", {
    excerpt: true,
    render: true,
  }).load();

  posts.sort(
    (a, b) =>
      +new Date(b.frontmatter.date as string) -
      +new Date(a.frontmatter.date as string)
  );

  for (const { url, excerpt, frontmatter, html } of posts) {
    feed.addItem({
      title: frontmatter.title,
      id: `${hostname}${url}`,
      link: `${hostname}${url}`,
      description: excerpt,
      author: [
        {
          name: "Victor Poughon",
          link: "https://victorpoughon.fr",
        },
      ],
      date: frontmatter.date,
    });
  }

  writeFileSync(path.join(config.outDir, "feed.rss"), feed.rss2());
}
