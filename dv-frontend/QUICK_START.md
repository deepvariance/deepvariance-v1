# Arayci Frontend - Quick Start

## Installation (3 Steps)

```bash
# 1. Install dependencies
npm install

# 2. Configure environment
cp .env.example .env

# 3. Start development server
npm run dev
```

**Access:** [http://localhost:3000](http://localhost:3000)

---

## Project Structure

```
src/
├── components/
│   ├── ui/              # shadcn/ui primitives (Button, Input, Table, etc.)
│   ├── layout/          # Layout components (AppShell, Sidebar, Navbar)
│   └── common/          # Shared components (StatusBadge)
│
├── features/
│   ├── datasets/        # Dataset management
│   ├── models/          # Model registry
│   ├── experiments/     # Experiment tracking
│   └── jobs/            # Job monitoring
│
├── lib/                 # Core utilities (API, store, utils)
├── hooks/               # Custom hooks (useTheme, usePagination, useModal)
├── routes/              # React Router configuration
├── types/               # TypeScript type definitions
├── App.tsx              # Root component
└── main.tsx             # Entry point
```

---

## Key Technologies

| Tech | Purpose |
|------|---------|
| React 18 | UI framework |
| TypeScript | Type safety |
| Vite | Build tool |
| Tailwind CSS | Styling |
| shadcn/ui | Component library |
| TanStack Table | Advanced tables |
| React Query | Data fetching |
| Zustand | State management |
| React Router | Routing |

---

## Available Pages

- `/datasets` - Dataset Explorer
- `/models` - Model Registry
- `/experiments` - Experiment Tracker
- `/jobs` - Job Monitor

---

## Environment Variables

```env
VITE_API_BASE_URL=https://api.deepvariance.ai/v1
VITE_ENV=development
VITE_APP_NAME=Arayci
```

---

## Commands

```bash
npm run dev      # Start dev server
npm run build    # Build for production
npm run preview  # Preview production build
npm run lint     # Run ESLint
npm run format   # Format with Prettier
```

---

## Features

- Dataset upload with drag-drop
- Model tracking and management
- Experiment monitoring
- Job status tracking
- Dark/Light theme toggle
- Responsive design
- Type-safe development
- Real-time data fetching

---

## Adding a New Feature

1. Create types in `src/types/`
2. Create hooks in `src/features/{feature}/hooks/`
3. Build components in `src/features/{feature}/components/`
4. Create page in `src/features/{feature}/pages/`
5. Add route in `src/routes/AppRoutes.tsx`

---

## Theme Colors

```css
Primary:  #00A7E1  /* DeepVariance blue */
Accent:   #38BDF8  /* Sky blue */
Success:  #22C55E  /* Green */
Error:    #EF4444  /* Red */
```

---

## Support

- **Documentation:** See [SETUP_GUIDE.md](SETUP_GUIDE.md)
- **Checklist:** See [IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md)
- **README:** See [README.md](README.md)

---

**Built by DeepVariance Team** © 2025
