export type JobStatus = 'queued' | 'processing' | 'completed' | 'failed';

export type ExtractionArtifact = {
  label: string;
  path: string;
  mimeType: string;
};

export type JobRecord = {
  id: string;
  filename: string;
  subject: string;
  status: JobStatus;
  progress: number;
  createdAt: string;
  updatedAt: string;
  preview: string | null;
  artifacts: ExtractionArtifact[];
  error: string | null;
};

type CreateJobInput = {
  filename: string;
  subject: string;
};

const jobs = new Map<string, JobRecord>();

function makeId() {
  return `job_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

function nowIso() {
  return new Date().toISOString();
}

function mockPreview(filename: string, subject: string) {
  return [
    `Extraction completed for ${filename}.`,
    `Subject: ${subject}.`,
    'This preview mirrors the existing Python pipeline output and will be replaced by the real worker response.',
  ].join(' ');
}

export function createJob(input: CreateJobInput) {
  const id = makeId();
  const timestamp = nowIso();
  const job: JobRecord = {
    id,
    filename: input.filename,
    subject: input.subject,
    status: 'queued',
    progress: 0,
    createdAt: timestamp,
    updatedAt: timestamp,
    preview: null,
    artifacts: [],
    error: null,
  };

  jobs.set(id, job);
  queueJob(id);
  return job;
}

export function getJob(id: string) {
  return jobs.get(id) ?? null;
}

export function listJobs() {
  return Array.from(jobs.values()).sort((left, right) => right.createdAt.localeCompare(left.createdAt));
}

function updateJob(id: string, updater: (job: JobRecord) => JobRecord) {
  const current = jobs.get(id);
  if (!current) {
    return null;
  }

  const next = updater({ ...current, artifacts: [...current.artifacts] });
  jobs.set(id, next);
  return next;
}

function queueJob(id: string) {
  setTimeout(() => {
    updateJob(id, (job) => ({
      ...job,
      status: 'processing',
      progress: 30,
      updatedAt: nowIso(),
    }));
  }, 650);

  setTimeout(() => {
    updateJob(id, (job) => ({
      ...job,
      status: 'processing',
      progress: 75,
      updatedAt: nowIso(),
    }));
  }, 1700);

  setTimeout(() => {
    updateJob(id, (job) => ({
      ...job,
      status: 'completed',
      progress: 100,
      updatedAt: nowIso(),
      preview: mockPreview(job.filename, job.subject),
      artifacts: [
        {
          label: 'Unified pretrain JSONL',
          path: `/artifacts/${job.id}/unified_pretrain.jsonl`,
          mimeType: 'application/jsonl',
        },
        {
          label: 'Unified SFT JSONL',
          path: `/artifacts/${job.id}/unified_sft.jsonl`,
          mimeType: 'application/jsonl',
        },
        {
          label: 'Dataset summary',
          path: `/artifacts/${job.id}/dataset_summary.json`,
          mimeType: 'application/json',
        },
      ],
    }));
  }, 3000);
}
