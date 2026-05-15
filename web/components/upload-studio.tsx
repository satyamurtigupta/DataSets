'use client';

import { useEffect, useMemo, useState } from 'react';
import type { JobRecord } from '@/lib/job-store';

type UploadResponse = {
  job: JobRecord;
  message: string;
};

const subjects = ['History', 'Geography', 'Science', 'Polity', 'Economics', 'Biology'];

export function UploadStudio() {
  const [file, setFile] = useState<File | null>(null);
  const [subject, setSubject] = useState('History');
  const [job, setJob] = useState<JobRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const canSubmit = Boolean(file) && !isSubmitting;

  useEffect(() => {
    if (!job || job.status === 'completed' || job.status === 'failed') {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const response = await fetch(`/api/jobs/${job.id}`);
        if (!response.ok) {
          return;
        }

        const data = (await response.json()) as { job: JobRecord };
        setJob(data.job);
      } catch {
        // Keep polling; transient network issues should not kill the UI.
      }
    }, 1200);

    return () => clearInterval(interval);
  }, [job]);

  const statusText = useMemo(() => {
    if (!job) {
      return 'Ready to upload a PDF.';
    }

    if (job.status === 'completed') {
      return 'Extraction completed and artifacts are ready.';
    }

    if (job.status === 'failed') {
      return 'The extraction job failed.';
    }

    if (job.status === 'processing') {
      return `Processing ${job.progress}%`;
    }

    return 'Upload received. Waiting in queue.';
  }, [job]);

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);

    if (!file) {
      setError('Choose a PDF file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('subject', subject);

    setIsSubmitting(true);

    try {
      const response = await fetch('/api/jobs', {
        method: 'POST',
        body: formData,
      });

      const payload = (await response.json()) as UploadResponse | { error: string };

      if (!response.ok) {
        setError('error' in payload ? payload.error : 'Upload failed.');
        return;
      }

      setJob((payload as UploadResponse).job);
    } catch {
      setError('Could not reach the extraction service.');
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <section className="studio-card" aria-label="Upload studio">
      <div className="studio-header">
        <div>
          <p className="eyebrow">Upload</p>
          <h2>Send a PDF to the extraction pipeline</h2>
        </div>
        <span className={`status-pill status-${job?.status ?? 'queued'}`}>{statusText}</span>
      </div>

      <form className="upload-form" onSubmit={onSubmit}>
        <label className="field">
          <span>Subject</span>
          <select value={subject} onChange={(event) => setSubject(event.target.value)}>
            {subjects.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>

        <label className="field file-field">
          <span>PDF file</span>
          <div className="file-picker">
            <input
              type="file"
              accept="application/pdf"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
            <p>{file ? file.name : 'Drop or choose a PDF. OCR-heavy files are supported.'}</p>
          </div>
        </label>

        <button className="primary-button" type="submit" disabled={!canSubmit}>
          {isSubmitting ? 'Launching job...' : 'Start extraction'}
        </button>
      </form>

      {error ? <p className="inline-error">{error}</p> : null}

      {job ? (
        <div className="job-panel">
          <div className="job-meta">
            <div>
              <p className="label">Job ID</p>
              <p>{job.id}</p>
            </div>
            <div>
              <p className="label">File</p>
              <p>{job.filename}</p>
            </div>
            <div>
              <p className="label">Progress</p>
              <p>{job.progress}%</p>
            </div>
          </div>

          <div className="progress-track" aria-hidden="true">
            <span style={{ width: `${job.progress}%` }} />
          </div>

          {job.preview ? (
            <div className="preview-block">
              <p className="label">Latest output</p>
              <p>{job.preview}</p>
            </div>
          ) : null}

          {job.artifacts.length > 0 ? (
            <div className="artifact-list">
              {job.artifacts.map((artifact) => (
                <div className="artifact-item" key={artifact.path}>
                  <div>
                    <p className="artifact-label">{artifact.label}</p>
                    <p>{artifact.path}</p>
                  </div>
                  <span>{artifact.mimeType}</span>
                </div>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}

      <p className="studio-footnote">
        This starter uses an in-memory job queue so the UI can be built and tested before the Python worker is wired in.
      </p>
    </section>
  );
}
