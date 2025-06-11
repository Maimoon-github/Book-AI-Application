# Large Files Management Guide

This document provides guidelines for handling large files in this repository.

## Current Setup

The repository has been configured to ignore the following large files and directories:

- Build artifacts (`app/build/`, `app/dist/`, `Book App/build/`, `Book App/dist/`)
- Executable files (`.exe`, `.pkg`, `.pyz`, `.dmg`, `.app`, `.deb`, `.rpm`)
- Media files (in `media/` and `uploads/` directories)

## GitHub Limitations

GitHub has the following file size limitations:
- 100MB is the maximum file size limit
- 50MB is the recommended maximum file size
- Large repositories with many large files may experience performance issues

## Best Practices for Large Files

1. **Avoid committing large files directly to the repository**
   - Build artifacts should be generated locally, not committed
   - Use CI/CD pipelines to build executables when needed

2. **For necessary large files, use Git LFS**
   - If you need to track large files, consider using [Git Large File Storage (LFS)](https://git-lfs.github.com/)
   - Installation instructions are available at [https://git-lfs.github.com/](https://git-lfs.github.com/)

3. **For PDF books and test data**
   - Consider using a separate storage solution for large datasets
   - Example options: AWS S3, Google Drive, Dropbox, etc.
   - Include download scripts in the repository instead

## Setting Up Git LFS (if needed)

```bash
# Install Git LFS
# Windows
git lfs install

# Track specific file types
git lfs track "*.pdf"
git lfs track "*.bin"

# Make sure .gitattributes is tracked
git add .gitattributes

# Add and commit files as usual
git add large_file.pdf
git commit -m "Add large file using Git LFS"
```

## Cleaning Up History (if large files are committed accidentally)

If large files are accidentally committed, you can use git-filter-repo to clean them up:

```bash
# Install git-filter-repo
pip install git-filter-repo

# Remove specific files
git filter-repo --path-glob "path/to/large/file" --invert-paths --force

# Push the cleaned repository
git push -f origin main
```

## Distribution Strategies

For distributing the application:

1. **Release builds via GitHub Releases**
   - Create tagged releases with release notes
   - Attach built executables as release assets (under 100MB)
   - For larger executables, provide download links to external storage

2. **Using installation scripts**
   - Include scripts that download necessary large files during installation
   - Example: `pip install -r requirements.txt && python download_resources.py`

## Additional Resources

- [GitHub documentation on large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files)
- [Git LFS documentation](https://git-lfs.github.com/spec/)
