# Arayci Frontend - Complete Setup Guide

## Project Overview

**Arayci** is DeepVariance's production-ready AutoML dashboard built with enterprise-grade technologies:

- React 18 + TypeScript for type-safe development
- Vite for blazing-fast builds
- shadcn/ui (Radix primitives) for accessible components
- TanStack Table for advanced data grids
- React Query for efficient server state
- Zustand for lightweight global state
- Tailwind CSS with custom DeepVariance theme

---

## Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:
```env
VITE_API_BASE_URL=https://api.deepvariance.ai/v1
VITE_ENV=development
VITE_APP_NAME=Arayci
```

### 3. Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

---

## Project Structure

```
arayci-frontend/
├── public/                      # Static assets
│   ├── favicon.svg             # App favicon
│   ├── robots.txt              # SEO
│   └── manifest.json           # PWA manifest
│
├── src/
│   ├── components/
│   │   ├── ui/                 # shadcn/ui primitives
│   │   │   ├── button.tsx
│   │   │   ├── input.tsx
│   │   │   ├── select.tsx
│   │   │   ├── checkbox.tsx
│   │   │   ├── dialog.tsx
│   │   │   ├── dropdown-menu.tsx
│   │   │   ├── table.tsx
│   │   │   ├── badge.tsx
│   │   │   ├── label.tsx
│   │   │   └── switch.tsx
│   │   │
│   │   ├── layout/             # Layout components
│   │   │   ├── AppShell.tsx    # Main layout wrapper
│   │   │   ├── Sidebar.tsx     # Navigation sidebar
│   │   │   ├── Navbar.tsx      # Top navbar with theme toggle
│   │   │   └── PageHeader.tsx  # Page header with actions
│   │   │
│   │   └── common/             # Shared components
│   │       └── StatusBadge.tsx # Status indicator component
│   │
│   ├── features/               # Feature modules
│   │   ├── datasets/
│   │   │   ├── pages/
│   │   │   │   └── DatasetExplorer.tsx
│   │   │   ├── components/
│   │   │   │   ├── DatasetTable.tsx
│   │   │   │   └── UploadDatasetModal.tsx
│   │   │   └── hooks/
│   │   │       └── useDatasets.ts
│   │   │
│   │   ├── models/
│   │   │   ├── pages/
│   │   │   │   └── ModelRegistry.tsx
│   │   │   ├── components/
│   │   │   │   └── ModelTable.tsx
│   │   │   └── hooks/
│   │   │       └── useModels.ts
│   │   │
│   │   ├── experiments/
│   │   │   ├── pages/
│   │   │   │   └── ExperimentTracker.tsx
│   │   │   └── hooks/
│   │   │       └── useExperiments.ts
│   │   │
│   │   └── jobs/
│   │       ├── pages/
│   │       │   └── JobMonitor.tsx
│   │       └── hooks/
│   │           └── useJobs.ts
│   │
│   ├── lib/                    # Core utilities
│   │   ├── api.ts              # Axios instance with interceptors
│   │   ├── queryClient.ts      # React Query configuration
│   │   ├── store.ts            # Zustand store (theme, sidebar)
│   │   ├── utils.ts            # Helper functions (cn, formatters)
│   │   └── constants.ts        # App constants and routes
│   │
│   ├── hooks/                  # Custom hooks
│   │   ├── useTheme.ts         # Theme management
│   │   ├── usePagination.ts    # Pagination logic
│   │   └── useModal.ts         # Modal state management
│   │
│   ├── routes/                 # Routing
│   │   └── AppRoutes.tsx       # Route definitions
│   │
│   ├── types/                  # TypeScript definitions
│   │   ├── common.ts           # Shared types
│   │   ├── dataset.ts          # Dataset types
│   │   ├── model.ts            # Model types
│   │   ├── experiment.ts       # Experiment types
│   │   └── job.ts              # Job types
│   │
│   ├── App.tsx                 # Root component
│   ├── main.tsx                # Entry point
│   └── index.css               # Global styles + Tailwind
│
├── .env.example                # Environment template
├── .eslintrc.cjs               # ESLint configuration
├── .prettierrc                 # Prettier configuration
├── components.json             # shadcn/ui config
├── package.json                # Dependencies
├── postcss.config.js           # PostCSS config
├── tailwind.config.ts          # Tailwind + DeepVariance theme
├── tsconfig.json               # TypeScript config
├── vite.config.ts              # Vite configuration
└── README.md                   # Project documentation
```

---

## Key Features Implemented

### 1. Layout System
- **AppShell**: Main layout wrapper with sidebar and navbar
- **Sidebar**: Collapsible navigation with icon-only mode
- **Navbar**: Top bar with theme toggle and user menu
- **PageHeader**: Reusable page header with action buttons

### 2. Dataset Explorer
- Upload datasets with drag-and-drop
- Table view with TanStack Table
- Status badges for dataset states
- Filter by status
- Delete functionality

### 3. Model Registry
- View trained models
- Status tracking (training, trained, deployed)
- Model metrics display
- Quick actions (train, delete)

### 4. Experiment Tracker
- Monitor ML experiments
- Track parameters and metrics
- Compare experiment results

### 5. Job Monitor
- View training and processing jobs
- Real-time status updates
- Job logs and error messages

### 6. Theme System
- Dark/Light mode toggle
- Persisted theme preference
- DeepVariance color palette
- CSS variables for easy customization

---

## Technology Details

### State Management

**Zustand Store** ([src/lib/store.ts](src/lib/store.ts))
- Theme state (dark/light)
- Sidebar state (open/collapsed)
- Persisted to localStorage

```typescript
const theme = useAppStore(state => state.theme)
const toggleTheme = useAppStore(state => state.toggleTheme)
```

### API Integration

**Axios Instance** ([src/lib/api.ts](src/lib/api.ts))
- Automatic auth token injection
- Request/response interceptors
- Centralized error handling

```typescript
import { api } from '@/lib/api'
const { data } = await api.get('/datasets')
```

### Data Fetching

**React Query** ([src/lib/queryClient.ts](src/lib/queryClient.ts))
- Automatic caching
- Background refetching
- Optimistic updates

```typescript
const { data, isLoading } = useDatasets(1, 10)
```

### Routing

**React Router v6** ([src/routes/AppRoutes.tsx](src/routes/AppRoutes.tsx))
- Nested routes
- Programmatic navigation
- Route protection (ready for auth)

---

## DeepVariance Theme

### Color System

```css
/* Dark Mode (default) */
--dv-primary: #00A7E1      /* Primary brand color */
--dv-bg: #0A0A0F           /* Background */
--dv-surface: #111827      /* Cards, panels */
--dv-accent: #38BDF8       /* Accents, highlights */
--dv-border: #1F2937       /* Borders */
--dv-text: #F9FAFB         /* Text */
--dv-success: #22C55E      /* Success states */
--dv-error: #EF4444        /* Error states */
```

### Typography

- **Primary Font**: Inter
- **Monospace**: Space Grotesk
- **Border Radius**: 0.75rem

---

## Component Library

### shadcn/ui Components

All components are built with:
- **Radix UI** primitives for accessibility
- **Tailwind CSS** for styling
- **Class Variance Authority** for variants
- Full TypeScript support

#### Available Components

1. **Button** - Primary actions
2. **Input** - Text input fields
3. **Select** - Dropdown selections
4. **Checkbox** - Binary choices
5. **Dialog** - Modal dialogs
6. **Dropdown Menu** - Context menus
7. **Table** - Data tables
8. **Badge** - Status indicators
9. **Label** - Form labels
10. **Switch** - Toggle switches

### Custom Components

**StatusBadge** ([src/components/common/StatusBadge.tsx](src/components/common/StatusBadge.tsx))
```tsx
<StatusBadge status="completed" />
```

---

## Development Workflow

### Adding a New Feature

1. **Create types** in `src/types/`
2. **Create API hooks** in `src/features/{feature}/hooks/`
3. **Build components** in `src/features/{feature}/components/`
4. **Create page** in `src/features/{feature}/pages/`
5. **Add route** in `src/routes/AppRoutes.tsx`

### Example: Adding a new "Pipelines" feature

```bash
# 1. Create directory structure
mkdir -p src/features/pipelines/{pages,components,hooks}

# 2. Create types
# src/types/pipeline.ts

# 3. Create hooks
# src/features/pipelines/hooks/usePipelines.ts

# 4. Create components
# src/features/pipelines/components/PipelineTable.tsx

# 5. Create page
# src/features/pipelines/pages/PipelineManager.tsx

# 6. Add route in AppRoutes.tsx
```

---

## API Integration

### Expected Backend Endpoints

```
GET    /datasets              - List datasets
POST   /datasets              - Create dataset
GET    /datasets/:id          - Get dataset
PATCH  /datasets/:id          - Update dataset
DELETE /datasets/:id          - Delete dataset

GET    /models                - List models
POST   /models                - Create model
GET    /models/:id            - Get model
PATCH  /models/:id            - Update model
DELETE /models/:id            - Delete model

GET    /experiments           - List experiments
POST   /experiments           - Create experiment
GET    /experiments/:id       - Get experiment

GET    /jobs                  - List jobs
POST   /jobs                  - Create job
GET    /jobs/:id              - Get job
```

### Response Format

```typescript
// Paginated response
{
  data: T[],
  total: number,
  page: number,
  pageSize: number,
  totalPages: number
}

// Single resource
{
  id: string,
  name: string,
  // ... other fields
}
```

---

## Scripts Reference

```bash
# Development
npm run dev          # Start dev server (port 3000)

# Production
npm run build        # Build for production
npm run preview      # Preview production build

# Code Quality
npm run lint         # Run ESLint
npm run format       # Format with Prettier
```

---

## Deployment

### Build for Production

```bash
npm run build
```

Output in `dist/` folder.

### Environment Variables for Production

```env
VITE_API_BASE_URL=https://api.production.com/v1
VITE_ENV=production
VITE_APP_NAME=Arayci
```

### Deploy to Vercel/Netlify

1. Connect repository
2. Set build command: `npm run build`
3. Set output directory: `dist`
4. Add environment variables

---

## Troubleshooting

### Port already in use
```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

### Module not found errors
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Type errors
```bash
# Regenerate TypeScript cache
rm -rf node_modules/.vite
npm run dev
```

---

## Next Steps

### Recommended Enhancements

1. **Authentication**
   - Add login/signup pages
   - JWT token management
   - Protected routes

2. **Advanced Tables**
   - Sorting and filtering
   - Column visibility toggle
   - Export to CSV

3. **Real-time Updates**
   - WebSocket integration
   - Live job status
   - Notifications

4. **Data Visualization**
   - Charts for metrics
   - Model performance graphs
   - Experiment comparison

5. **Testing**
   - Unit tests with Vitest
   - Component tests with Testing Library
   - E2E tests with Playwright

---

## Support

For questions or issues:
- GitHub Issues: [repository-url]/issues
- Documentation: See README.md
- Team: DeepVariance Engineering

---

Built with by **DeepVariance Team** © 2025
