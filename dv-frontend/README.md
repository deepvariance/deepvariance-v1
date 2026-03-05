# DeepVariance Frontend

**DeepVariance** is a research-first machine learning platform for model development and dataset management, built with React + Vite + TypeScript.

## Features

- **Dataset Management**: Upload, explore, and manage datasets
- **Model Registry**: Track and manage ML models
- **Experiment Tracking**: Monitor and compare experiments
- **Job Monitoring**: Track training and processing jobs
- **Dark/Light Theme**: Toggle between dark and light modes
- **Responsive Design**: Works seamlessly across devices

## Tech Stack

- **React 18** - Modern React with hooks
- **TypeScript** - Type-safe development
- **Vite** - Lightning-fast build tool
- **Tailwind CSS** - Utility-first CSS framework
- **shadcn/ui** - Radix UI primitives + Tailwind
- **TanStack Table** - Powerful data tables
- **React Query** - Server state management
- **Zustand** - Global state management
- **React Hook Form** - Form handling
- **React Router v7** - Client-side routing
- **Lucide Icons** - Beautiful icon library

## Getting Started

### Prerequisites

- Node.js 18+ and npm

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd dv-frontend
```

2. Install dependencies:
```bash
npm install
```

3. Set up environment variables:
```bash
cp .env.example .env
```

Edit `.env` to configure your API endpoint:
```env
VITE_API_BASE_URL=https://api.deepvariance.ai/v1
VITE_ENV=development
VITE_APP_NAME=DeepVariance
```

4. Start the development server:
```bash
npm run dev
```

The app will be available at [http://localhost:3000](http://localhost:3000)

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint
- `npm run format` - Format code with Prettier

## Project Structure

```
src/
├── components/
│   ├── ui/                 # shadcn UI primitives
│   ├── layout/             # Layout components (AppShell, Sidebar, Navbar)
│   └── common/             # Shared components (StatusBadge)
│
├── features/
│   ├── datasets/           # Dataset feature module
│   ├── models/             # Model feature module
│   ├── experiments/        # Experiment feature module
│   └── jobs/               # Job feature module
│
├── lib/
│   ├── api.ts              # Axios instance
│   ├── queryClient.ts      # React Query client
│   ├── store.ts            # Zustand store
│   ├── utils.ts            # Utility functions
│   └── constants.ts        # App constants
│
├── hooks/                  # Custom React hooks
├── routes/                 # Route definitions
├── types/                  # TypeScript type definitions
├── App.tsx                 # Main App component
└── main.tsx                # Entry point
```

## Design System

### Color Palette

| Token | Value | Usage |
|-------|-------|-------|
| `--dv-primary` | `#00A7E1` | Primary brand color |
| `--dv-bg` | `#0A0A0F` | Background (dark mode) |
| `--dv-surface` | `#111827` | Card/Surface color |
| `--dv-accent` | `#38BDF8` | Accent color |
| `--dv-border` | `#1F2937` | Border color |
| `--dv-text` | `#F9FAFB` | Text color |
| `--dv-success` | `#22C55E` | Success state |
| `--dv-error` | `#EF4444` | Error state |

### Typography

- **Font**: Inter, Space Grotesk
- **Border Radius**: 0.75rem

## Features Checklist

- [x] Project setup with Vite + React + TypeScript
- [x] Tailwind CSS configuration with DeepVariance theme
- [x] shadcn/ui components integration
- [x] Layout components (Sidebar, Navbar, AppShell)
- [x] React Router setup
- [x] Zustand global state management
- [x] React Query configuration
- [x] Dataset Explorer page with table and filters
- [x] Model Registry page
- [x] Experiment Tracker page
- [x] Job Monitor page
- [x] Dark/Light theme toggle
- [x] TypeScript type definitions
- [x] ESLint + Prettier configuration

## API Integration

The app is configured to connect to the DeepVariance API. Update the `VITE_API_BASE_URL` in your `.env` file to point to your backend.

Example API endpoints:
- `GET /datasets` - List datasets
- `POST /datasets` - Create dataset
- `GET /models` - List models
- `GET /experiments` - List experiments
- `GET /jobs` - List jobs

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m "Add my feature"`
4. Push to branch: `git push origin feature/my-feature`
5. Submit a pull request

## License

© 2025 DeepVariance. All rights reserved.

## Support

For issues and questions, please open an issue on the GitHub repository.

---

Built with by DeepVariance Team
