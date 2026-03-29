# Quick Guide: Compiling main_enhanced.tex

## ✅ Prerequisites

1. **LaTeX Distribution** installed (MiKTeX, TeX Live, or MacTeX)
2. **Figure files** in the `latex/` directory

## 📁 Step 1: Copy Figures

Make sure you have the plot figures in your latex directory:

```powershell
# If figures are in reports/figures/
Copy-Item d:\Work\UAV\reports\figures\*.png d:\Work\UAV\latex\

# Verify they exist
dir d:\Work\UAV\latex\*.png
```

You need these 4 files:
- ✅ `reward_curve.png`
- ✅ `secrecy_bar.png`
- ✅ `cdf_rates.png`
- ✅ `pareto.png`

## 🔧 Step 2: Compile LaTeX

### Option A: Using Command Line

```powershell
cd d:\Work\UAV\latex

# First pass (generates aux file)
pdflatex main_enhanced.tex

# Process bibliography
bibtex main_enhanced

# Second pass (includes citations)
pdflatex main_enhanced.tex

# Third pass (resolves cross-references)
pdflatex main_enhanced.tex
```

### Option B: Using VS Code with LaTeX Workshop Extension

1. Open `main_enhanced.tex` in VS Code
2. Press `Ctrl+Alt+B` to build
3. Press `Ctrl+Alt+V` to view PDF

### Option C: Using TeXstudio/TeXworks

1. Open `main_enhanced.tex`
2. Select `pdfLaTeX` as compiler
3. Press F5 (or click green arrow)
4. Run BibTeX (Tools → Bibliography)
5. Press F5 twice more

## 📊 Step 3: Verify Output

The output `main_enhanced.pdf` should have:

✅ **8 pages** (approximately)  
✅ **4 figures** (all rendering correctly)  
✅ **4 tables** (formatted properly)  
✅ **8 references** (numbered [1]-[8])  
✅ **24 equations** (numbered and aligned)  
✅ **IEEE two-column format**  

## 🐛 Troubleshooting

### Problem: "File not found: reward_curve.png"
**Solution**: Copy figures to latex directory (see Step 1)

### Problem: "Undefined citations"
**Solution**: Run BibTeX, then pdflatex twice more

### Problem: "Package siunitx not found"
**Solution**: 
```powershell
# MiKTeX
mpm --install=siunitx

# TeX Live
tlmgr install siunitx
```

### Problem: "Overfull hbox" warnings
**Solution**: These are minor formatting warnings, PDF still generates correctly

### Problem: Missing figures in output
**Solution**: Check that PNG files are in same directory as .tex file

## 📝 Quick Comparison Commands

Compare with original paper:

```powershell
# Original
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Enhanced
pdflatex main_enhanced.tex
bibtex main_enhanced
pdflatex main_enhanced.tex
pdflatex main_enhanced.tex

# Open both PDFs side-by-side
start main.pdf
start main_enhanced.pdf
```

## ✅ Expected Differences

| Aspect | main.pdf | main_enhanced.pdf |
|--------|----------|-------------------|
| **Pages** | ~6 | ~8 |
| **Equations** | 5 | 24 |
| **Tables** | 2 | 4 |
| **References** | 3 | 8 |
| **Subsections** | 7 | 17 |
| **Technical Depth** | Basic | Advanced |

## 🎯 Final Checklist

Before submission:

- [ ] All figures render correctly
- [ ] All equations are numbered
- [ ] All citations appear as [1], [2], etc.
- [ ] No compilation errors
- [ ] No missing references
- [ ] Table data matches your results
- [ ] Author names and IDs correct
- [ ] Mentor name correct (Dr. Sunandita Debnath)

## 📤 Submission Files

For conference/journal submission, prepare:

```
submission/
├── main_enhanced.pdf          # Final PDF
├── main_enhanced.tex          # Source file
├── IEEEtran.cls              # IEEE class file
├── reward_curve.png
├── secrecy_bar.png
├── cdf_rates.png
├── pareto.png
└── README.txt                # Compilation instructions
```

## 🎓 Ready to Submit!

Once compiled successfully, your paper is **publication-ready** for:

✅ IEEE Transactions on Wireless Communications  
✅ IEEE Transactions on Vehicular Technology  
✅ IEEE GLOBECOM / VTC conferences  
✅ IEEE Communications Letters  

**Your enhanced paper now properly documents all the IEEE-worthy models you've implemented!** 🚀📝
