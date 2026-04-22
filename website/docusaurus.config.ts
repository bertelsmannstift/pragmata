import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'pRAGmata',
  tagline: 'Evidence-grounded evaluation for RAG systems',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  url: 'https://bertelsmannstift.github.io',
  baseUrl: '/pragmata/',

  organizationName: 'bertelsmannstift',
  projectName: 'pragmata',

  onBrokenLinks: 'throw',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          path: 'docs',
          routeBasePath: 'docs',
          sidebarPath: './sidebars.ts',
          editUrl:
            'https://github.com/bertelsmannstift/pragmata/tree/main/website/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  plugins: [
  [
    "@docusaurus/plugin-content-docs",
    {
      id: "api",
      path: "api",
      routeBasePath: "api",
      sidebarPath: "./sidebarsApi.ts",
      editUrl:
            'https://github.com/bertelsmannstift/pragmata/tree/main/website/',
    },
  ],
  [
    "@docusaurus/plugin-content-docs",
    {
      id: "community",
      path: "community",
      routeBasePath: "community",
      sidebarPath: "./sidebarsCommunity.ts",
      editUrl:
            'https://github.com/bertelsmannstift/pragmata/tree/main/website/',
    },
  ],
],

  themeConfig: {
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'pRAGmata',
      items: [
        {to: "/docs/introduction", label: "Docs", position: "left"},
        {to: "/api/overview", label: "API", position: "left"},
        {to: "/community/team", label: "Community", position: "left"},
        {
          href: 'https://github.com/bertelsmannstift/pragmata',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Learn',
          items: [
            {label: 'Introduction', to: '/docs/introduction'},
            {label: 'Getting started', to: '/docs/getting-started/installation'},
          ],
        },
        {
          title: 'Community',
          items: [
            {label: 'Team', to: '/community/team'},
            {label: 'Contributing', to: '/community/contributing'},
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/bertelsmannstift/pragmata',
            },
          ],
        },
      ],
       copyright:
           `Copyright © ${new Date().getFullYear()} pRAGmata contributors`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
