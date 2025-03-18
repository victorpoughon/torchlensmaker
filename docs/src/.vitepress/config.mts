import { defineConfig } from 'vitepress'

// Vite plugins
import version from "vite-plugin-package-version";

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Torch Lens Maker",
  description: "Differentiable geometric optics in PyTorch",
  head: [['link', { rel: 'icon', href: '/logos/tlmlogo_black130_margin.png' }]],
  themeConfig: {
    logo: '/logos/tlmlogo_black150.png',
    notFound: {
      quote: "Sorry!"
    },

    nav: [
      { text: 'Documentation', link: '/' },
      { text: 'About', link: '/about' },
      { text: 'Roadmap', link: '/roadmap' },
      {
        text: 'Community',
        items: [
          { text: 'GitHub Discussions', link: 'https://github.com/victorpoughon/torchlensmaker/discussions' },
          { text: 'Funding', link: '/about#funding' },
          { text: 'Mailing List', link: '/about#newsletter' }
        ]
      }
    ],

    sidebar: [
      {
        text: 'Introduction',
        items: [
          { text: 'Welcome', link: '/'},
          { text: 'Design Overview', link: '/design-overview' },
          { text: 'Installation', link: '/installation' },
          { text: 'Getting Started', link: '/getting-started' },
        ]
      },
      {
        text: 'Examples',
        items: [
          { text: 'Cooke Triplet', link: '/examples/cooke_triplet'},
          { text: 'Double Gauss', link: '/examples/double_gauss'},
          { text: 'Landscape Lens', link: '/examples/landscape'},
          { text: 'Magnifying glass', link: '/examples/magnifying_glass'},
          { text: 'Pinhole Camera', link: '/examples/pinhole_camera'},
          { text: 'Pink Floyd', link: '/examples/pink_floyd'},
          { text: 'Rainbow', link: '/examples/rainbow'},
          { text: 'Reflecting Telescope', link: '/examples/reflecting_telescope'},
          { text: 'Snell\'s Window', link: '/examples/snells_window'},
          { text: 'Triple Biconvex', link: '/examples/triple_biconvex'},
          { text: 'Variable Lens Sequence', link: '/examples/variable_lens_sequence'},
          { text: 'Test notebooks', link: '/test_notebooks'}
        ]
      },

      {
        text: 'Advanced Topics',
        items: [
          { text: 'Collision detection', link: '/advanced/collision_detection'}
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/victorpoughon/torchlensmaker' }
    ],

    search: {
      provider: 'local'
    }
  },

  cleanUrls: true,

  markdown: {
    math: true,
    defaultHighlightLang: "python",
    theme : {
      light : 'github-light',
      dark: 'github-dark',
  }},

  vue: {
    template: {
      transformAssetUrls: {
        video: ['src', 'poster'],
        source: ['src'],
        img: ['src'],
        image: ['xlink:href', 'href'],
        use: ['xlink:href', 'href'],
        TLMViewer: ['src'],
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
  },
})
