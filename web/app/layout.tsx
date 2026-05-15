import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'PaperPilot',
  description: 'A SaaS for uploading PDFs and extracting clean dataset artifacts.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
