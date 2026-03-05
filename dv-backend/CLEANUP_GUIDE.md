# 🎯 DeepVariance - Clean Directory Guidelines

## ✅ Implemented Changes

### 1. Enhanced `.gitignore`

Comprehensive gitignore patterns now cover:

- ✓ AutoGluon model directories (`autogluon_models_*/`)
- ✓ All ML model formats (`.pkl`, `.joblib`, `.pth`, `.h5`, etc.)
- ✓ Generated model code files
- ✓ Training artifacts and checkpoints
- ✓ Test datasets and temporary files
- ✓ All result directories with UUID names

### 2. Directory Structure Documentation

Created comprehensive structure docs:

- ✓ `DIRECTORY_STRUCTURE.md` - Complete backend architecture
- ✓ `data/README.md` - Dataset storage guide
- ✓ `models/README.md` - Model storage guide
- ✓ `results/README.md` - Results storage guide

### 3. Git-Tracked Preservation

Each storage directory now has:

- ✓ `.gitkeep` - Preserves empty directories in git
- ✓ `README.md` - Documents directory purpose

## 📊 Storage Directories

### Currently Gitignored (Generated Assets)

```
✗ autogluon_models_1771206748/   (14 directories total)
✗ models/*.pkl                    (4 files)
✗ models/generated_model_*.py     (37 files)
✗ data/{uuid}/                    (7 dataset directories)
✗ results/{uuid}/                 (40 result directories)
✗ logs/*.log                      (all log files)
```

### Git-Tracked (Structure Only)

```
✓ data/.gitkeep
✓ data/README.md
✓ models/.gitkeep
✓ models/README.md
✓ results/.gitkeep
✓ results/README.md
✓ DIRECTORY_STRUCTURE.md
```

## 🧹 Cleanup Recommendations

### Immediate Actions

```bash
# Navigate to backend
cd /Users/saaivigneshp/Desktop/dv-backend

# Remove all AutoGluon artifacts (saves ~2GB+)
rm -rf autogluon_models_*

# Clean old generated models
cd models
rm -f generated_model_*.py best_model_*.py *.pkl

# Clean old test data
cd ../data
rm -rf test_automl_*.csv test_dataset/

# Clean old results
cd ../results
find . -type d -mindepth 1 -maxdepth 1 ! -name '.gitkeep' -exec rm -rf {} +

# Verify cleanup
cd ..
du -sh data/ models/ results/
```

### Periodic Maintenance

Run monthly or after major testing:

```bash
# Check sizes
du -sh autogluon_models_*/ models/ data/ results/ logs/

# Archive old logs
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/*.log
rm logs/*.log

# Remove models older than 30 days
find models/ -name "*.pkl" -mtime +30 -delete
find models/ -name "generated_model_*.py" -mtime +30 -delete
```

## 📏 Size Guidelines

### Healthy Sizes

- `data/`: < 1GB (depends on datasets)
- `models/`: < 500MB (archived models)
- `results/`: < 200MB (recent runs only)
- `logs/`: < 50MB (rotate regularly)

### Warning Signs

- AutoGluon folders: Each can be 500MB-2GB
- Multiple `autogluon_models_*` directories accumulating
- 40+ result directories (current state)
- 30+ generated model files (current state)

## 🔒 Security Best Practices

### Never Commit

- ❌ `.env` file (contains secrets)
- ❌ Database files (`db/`, `*.sqlite`)
- ❌ Trained models (can contain data leakage)
- ❌ User-uploaded datasets
- ❌ API keys or credentials

### Always Commit

- ✅ `.gitkeep` files
- ✅ `README.md` documentation
- ✅ `.env.example` (template only)
- ✅ Code files (`.py`, `.ts`, `.tsx`)
- ✅ Configuration templates

## 🎯 Directory Ownership

### Backend Responsibilities

| Directory             | Purpose          | Owner     | Cleanup Freq |
| --------------------- | ---------------- | --------- | ------------ |
| `data/`               | Dataset storage  | DatasetDB | On delete    |
| `models/`             | Model files      | ModelDB   | On delete    |
| `results/`            | Training outputs | JobDB     | Manual/30d   |
| `autogluon_models_*/` | AutoML cache     | AutoGluon | After job    |
| `logs/`               | Application logs | Logger    | Weekly       |

### Git Repository Structure

```
dv-backend/           (git tracked)
├── src files         ✓ Tracked
├── config files      ✓ Tracked
├── .gitkeep files    ✓ Tracked
├── README docs       ✓ Tracked
└── data/models/      ✗ Gitignored (content only)
    results/          ✗ Gitignored (content only)
    logs/             ✗ Gitignored
```

## 🚀 Deployment Checklist

Before deploying:

- [ ] Verify `.env` is properly configured
- [ ] Ensure `.gitignore` is comprehensive
- [ ] Remove all test/development data
- [ ] Check no credentials in git history
- [ ] Verify directory permissions
- [ ] Test with empty data/models/results dirs

After deployment:

- [ ] Monitor disk usage
- [ ] Set up log rotation
- [ ] Configure automatic cleanup jobs
- [ ] Implement storage quotas

## 📞 Quick Reference

**Check ignored files:**

```bash
git status --ignored
```

**See what would be cleaned:**

```bash
git clean -ndX
```

**Force clean all gitignored files:**

```bash
git clean -fdX  # BE CAREFUL - PERMANENT
```

**Disk usage:**

```bash
du -sh autogluon_models_*/ data/ models/ results/ logs/
```
