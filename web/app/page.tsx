import { UploadStudio } from '@/components/upload-studio';

const highlights = [
  {
    title: 'Python pipeline as source of truth',
    text: 'The existing scripts remain the core extractor while the SaaS adds upload, job tracking, and delivery.',
  },
  {
    title: 'Async by default',
    text: 'OCR-heavy PDFs queue as jobs so the UI stays fast and failures are visible.',
  },
  {
    title: 'API-first outputs',
    text: 'Each job is built to expose the existing JSONL and summary artifacts through a clean contract.',
  },
];

const apiSteps = [
  'POST /api/jobs to upload a PDF and create a job.',
  'GET /api/jobs/:jobId to poll status and progress.',
  'Download generated artifacts once the job is complete.',
];

export default function HomePage() {
  return (
    <main className="page-shell">
      <div className="ambient ambient-a" />
      <div className="ambient ambient-b" />

      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">PaperPilot</p>
          <h1>Turn complex PDFs into a productized extraction service.</h1>
          <p className="hero-text">
            Upload a PDF, let the worker process clean pages and OCR fallbacks, then serve the extracted outputs through a polished SaaS dashboard and API.
          </p>

          <div className="hero-metrics">
            <div>
              <strong>Async jobs</strong>
              <span>Queue first, extract second</span>
            </div>
            <div>
              <strong>Existing scripts</strong>
              <span>No pipeline rewrite</span>
            </div>
            <div>
              <strong>API-ready</strong>
              <span>Downloadable JSONL outputs</span>
            </div>
          </div>
        </div>

        <UploadStudio />
      </section>

      <section className="content-grid">
        {highlights.map((item) => (
          <article className="feature-card" key={item.title}>
            <p className="eyebrow">Foundation</p>
            <h2>{item.title}</h2>
            <p>{item.text}</p>
          </article>
        ))}
      </section>

      <section className="api-panel">
        <div>
          <p className="eyebrow">API contract</p>
          <h2>Shape the backend around the extractor, not the other way around.</h2>
          <p>
            The first release should accept uploads, return a job ID, and expose the same JSONL outputs your Python scripts already generate.
          </p>
        </div>
        <ol className="api-list">
          {apiSteps.map((step) => (
            <li key={step}>{step}</li>
          ))}
        </ol>
      </section>
    </main>
  );
}
