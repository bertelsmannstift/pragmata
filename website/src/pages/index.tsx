import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();

  return (
    <Layout title={siteConfig.title} description={siteConfig.tagline}>
      <main className="container margin-vert--xl">
        <div style={{maxWidth: 720}}>
          <Heading as="h1">{siteConfig.title}</Heading>
          <p>{siteConfig.tagline}</p>
          <p>
            pRAGmata is a Python framework for empirically evaluating retrieval-augmented generation (RAG) systems.
          </p>
          <div style={{display: 'flex', gap: '1rem', flexWrap: 'wrap'}}>
            <Link className="button button--primary button--lg" to="/docs/introduction">
              Read the docs
            </Link>
            <Link className="button button--secondary button--lg" to="/api/overview">
              Explore the API
            </Link>
          </div>
        </div>
      </main>
    </Layout>
  );
}