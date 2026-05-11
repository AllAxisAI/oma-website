import { useState, useEffect } from 'react';
import Landing from './pages/Landing';
import DocsLayout from './pages/docs/DocsLayout';
import DocsHome from './pages/docs/DocsHome';
import GettingStarted from './pages/docs/GettingStarted';
import MethodSystem from './pages/docs/concepts/MethodSystem';
import LossSystem from './pages/docs/concepts/LossSystem';
import DataPipeline from './pages/docs/concepts/DataPipeline';
import CustomMethod from './pages/docs/guides/CustomMethod';
import CustomLoss from './pages/docs/guides/CustomLoss';
import Logging from './pages/docs/guides/Logging';
import Evaluation from './pages/docs/guides/Evaluation';
import DiffusionBridgeRecipe from './pages/docs/recipes/DiffusionBridgeRecipe';
import DiffusionSystem from './pages/docs/concepts/DiffusionSystem';


function useHashRoute(): string {
  const getPath = () => {
    const hash = window.location.hash;
    return hash ? hash.slice(1) : '/';
  };
  const [route, setRoute] = useState(getPath);
  useEffect(() => {
    const handler = () => setRoute(getPath());
    window.addEventListener('hashchange', handler);
    return () => window.removeEventListener('hashchange', handler);
  }, []);
  return route;
}

const DOCS_PAGES: Record<string, React.ComponentType> = {
  '/docs': DocsHome,
  '/docs/getting-started': GettingStarted,
  '/docs/concepts/method-system': MethodSystem,
  '/docs/concepts/loss-system': LossSystem,
  '/docs/concepts/data-pipeline': DataPipeline,
  '/docs/guides/custom-method': CustomMethod,
  '/docs/guides/custom-loss': CustomLoss,
  '/docs/guides/logging': Logging,
  '/docs/guides/evaluation': Evaluation,
  '/docs/recipes/diffusion-bridge': DiffusionBridgeRecipe,
  '/docs/concepts/diffusion-system': DiffusionSystem,
};

export default function App() {
  const route = useHashRoute();

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [route]);

  const DocsPage = DOCS_PAGES[route];
  if (DocsPage) {
    return (
      <DocsLayout currentRoute={route}>
        <DocsPage />
      </DocsLayout>
    );
  }
  return <Landing />;
}
