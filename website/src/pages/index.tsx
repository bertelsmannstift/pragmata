import type { ReactNode } from "react";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";
import Heading from "@theme/Heading";

import styles from "./index.module.css";

export default function Home(): ReactNode {
  const { siteConfig } = useDocusaurusContext();

  return (
    <Layout title={siteConfig.title} description={siteConfig.tagline}>
      <main className={styles.hero}>
        <div className="container">
          <div className={styles.inner}>
            <Heading as="h1" className={styles.title}>
              pRAGmata Docs
            </Heading>

            <p className={styles.lead}>
              Documentation for the pRAGmata Python framework.
            </p>

            <p className={styles.description}>
              Use <strong>pRAGmata querygen</strong> for spec-driven synthetic
              query generation and <strong>pRAGmata annotation</strong> for
              web-based annotation to label query-response pairs.
            </p>

            <div className={styles.actions}>
              <Link
                className="button button--primary button--lg"
                to="/docs/introduction"
              >
                Read the docs
              </Link>
              <Link
                className="button button--secondary button--lg"
                to="/api/overview"
              >
                Explore the API
              </Link>
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}