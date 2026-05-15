import { NextRequest, NextResponse } from 'next/server';
import { createJob, listJobs } from '@/lib/job-store';

export async function GET() {
  return NextResponse.json({ jobs: listJobs() });
}

export async function POST(request: NextRequest) {
  const formData = await request.formData();
  const file = formData.get('file');
  const subject = String(formData.get('subject') ?? 'General').trim() || 'General';

  if (!(file instanceof File)) {
    return NextResponse.json({ error: 'A PDF file is required.' }, { status: 400 });
  }

  if (file.size === 0) {
    return NextResponse.json({ error: 'The uploaded file is empty.' }, { status: 400 });
  }

  const job = createJob({
    filename: file.name,
    subject,
  });

  return NextResponse.json(
    {
      job,
      message: 'Upload accepted. Extraction has been queued.',
    },
    { status: 201 },
  );
}
