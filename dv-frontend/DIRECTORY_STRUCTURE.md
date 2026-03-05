# Frontend Directory Structure

## 📁 Application Structure

```
dv-frontend/
├── index.html                   # Entry HTML
├── package.json                 # Dependencies
├── vite.config.ts              # Vite configuration
├── tsconfig.json               # TypeScript config
└── src/
    ├── main.tsx                # Application entry point
    ├── app/
    │   ├── App.tsx             # Root component
    │   ├── globals.css         # Global styles
    │   └── routes.tsx          # Route definitions
    ├── features/               # Feature modules
    │   ├── datasets/
    │   │   ├── DatasetsPage.tsx
    │   │   ├── DatasetDetailPage.tsx
    │   │   └── components/
    │   ├── models/
    │   │   ├── ModelsPage.tsx
    │   │   ├── ModelDetailPage.tsx
    │   │   └── components/
    │   ├── training/
    │   │   ├── TrainingRunnerPage.tsx
    │   │   └── components/
    │   ├── analytics/
    │   │   ├── ModelPerformancePage.tsx
    │   │   └── UsageMetricsPage.tsx
    │   └── home/
    │       └── HomePage.tsx
    ├── shared/
    │   ├── api/                # API client layer
    │   │   ├── client.ts       # Axios instance
    │   │   ├── datasets.ts
    │   │   ├── models.ts
    │   │   └── jobs.ts
    │   ├── hooks/              # React Query hooks
    │   │   ├── useDatasets.ts
    │   │   ├── useModels.ts
    │   │   └── useJobs.ts
    │   ├── components/         # Shared UI components
    │   ├── types/              # TypeScript definitions
    │   ├── config/             # App configuration
    │   └── utils/              # Utility functions
    └── assets/                 # Static assets
```

## 🎨 Feature Module Pattern

Each feature follows this structure:

```
features/{feature}/
├── {Feature}Page.tsx           # Main page component
├── {Feature}DetailPage.tsx     # Detail view (if applicable)
└── components/                 # Feature-specific components
    ├── {Feature}Table.tsx
    ├── {Feature}Form.tsx
    └── {Feature}Card.tsx
```

## 🔌 State Management

### Server State (React Query)

- All API data managed by React Query
- Hooks in `shared/hooks/`
- Automatic caching and invalidation

### UI State (Local)

- Component-local state with `useState`
- No global state management needed for most cases

## 📦 Build Artifacts (gitignored)

```
node_modules/                   # Dependencies
dist/                          # Production build
dist-ssr/                      # SSR build
.vscode/                       # VS Code settings
logs/                          # Development logs
```

## 🚀 Development

```bash
# Install dependencies
npm install

# Start dev server (port 3000)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## 📝 Conventions

1. **Components**: PascalCase, functional components
2. **Hooks**: camelCase, prefixed with `use`
3. **Types**: PascalCase, in `shared/types/`
4. **API calls**: Never directly in components, always through hooks
5. **Styling**: Mantine v7 components + custom CSS

## 🔗 API Integration

Pattern: Component → Hook → API → Backend

```typescript
// Component
const { data: datasets } = useDatasets()

// Hook (shared/hooks/useDatasets.ts)
export const useDatasets = () =>
  useQuery({
    queryKey: ['datasets'],
    queryFn: getDatasets,
  })

// API (shared/api/datasets.ts)
export const getDatasets = async () => {
  const { data } = await apiClient.get('/datasets')
  return data
}
```
