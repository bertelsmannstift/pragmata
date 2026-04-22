import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'introduction',
    {
      type: 'category',
      label: 'Getting started',
      items: ['getting-started/why-use-pragmata', 'getting-started/installation'],
    },
    {
      type: 'category',
      label: 'Guides',
      items: ['guides/querygen', 'guides/annotation'],
    },
  ],
};

export default sidebars;
