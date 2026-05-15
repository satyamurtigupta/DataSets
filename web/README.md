# PaperPilot Web

Next.js frontend for the PDF extraction SaaS.

## Local development

```bash
npm install
npm run dev
```

The first implementation uses an in-memory job store and mock extraction progress so the upload flow and dashboard can be developed before wiring the Python worker.
