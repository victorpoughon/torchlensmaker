import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Torch Lens Maker",
  description: "Differentiable geometric optics in PyTorch",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Examples', link: '/markdown-examples' },
      { text: 'About', link: '/about' }
    ],

    sidebar: [
      {
        text: 'Examples',
        items: [
          { text: 'Markdown Examples', link: '/markdown-examples' },
          { text: 'Runtime API Examples', link: '/api-examples' },
          { text: 'Design Overview', link: '/design' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/victorpoughon/torchlensmaker' }
    ]
  },

  markdown: {
    math: true,
    defaultHighlightLang: "python",
    theme : {
      light : 'github-light',
      dark: 'github-dark',
  }}
})
