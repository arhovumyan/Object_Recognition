#!/bin/bash

# Script to push the drone object recognition system to GitHub

echo "🚁 Drone Object Recognition - GitHub Push Helper"
echo "=================================================="

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: Not in a git repository"
    echo "Run this script from the ObjectRec directory"
    exit 1
fi

# Check if remote origin exists
if ! git remote get-url origin >/dev/null 2>&1; then
    echo "📝 Setting up GitHub remote..."
    echo ""
    echo "Please create a new repository on GitHub first:"
    echo "1. Go to https://github.com/new"
    echo "2. Repository name: drone-object-recognition"
    echo "3. Description: ROS2-based drone object recognition system using YOLOv8s and MobileNetV3"
    echo "4. Make it Public or Private (your choice)"
    echo "5. DON'T initialize with README (we already have one)"
    echo ""
    read -p "Enter your GitHub username: " github_username
    read -p "Enter repository name [drone-object-recognition]: " repo_name
    repo_name=${repo_name:-drone-object-recognition}
    
    echo "🔗 Adding remote origin..."
    git remote add origin "https://github.com/${github_username}/${repo_name}.git"
else
    echo "✅ Remote origin already configured"
fi

# Show current status
echo ""
echo "📊 Repository Status:"
echo "===================="
git status --short

# Check if there are uncommitted changes
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo ""
    echo "⚠️  You have uncommitted changes!"
    echo "Do you want to commit them now? (y/n)"
    read -p "> " commit_choice
    if [[ $commit_choice =~ ^[Yy]$ ]]; then
        echo "📝 Committing changes..."
        git add .
        read -p "Enter commit message: " commit_msg
        git commit -m "$commit_msg"
    else
        echo "❌ Please commit your changes first"
        exit 1
    fi
fi

# Push to GitHub
echo ""
echo "🚀 Pushing to GitHub..."
echo "======================="

# Try to push
if git push -u origin master; then
    echo ""
    echo "🎉 Successfully pushed to GitHub!"
    echo "================================="
    echo ""
    echo "Your repository is now available at:"
    git remote get-url origin
    echo ""
    echo "📋 What's included:"
    echo "✅ Complete ROS2 drone object recognition system"
    echo "✅ YOLOv8s + MobileNetV3 detection pipeline"
    echo "✅ High-quality video recording capabilities"
    echo "✅ Comprehensive documentation and guides"
    echo "✅ Automated setup scripts"
    echo "✅ Analysis and visualization tools"
    echo ""
    echo "🔧 For users to get started:"
    echo "1. git clone $(git remote get-url origin)"
    echo "2. cd drone-object-recognition"
    echo "3. ./setup.sh"
    echo ""
    echo "📚 Documentation available:"
    echo "- README.md - Main project overview"
    echo "- USAGE_GUIDE.md - Detailed usage instructions"
    echo "- HIGH_QUALITY_GUIDE.md - Video recording setup"
    echo "- GITHUB_SETUP.md - Setup guide for GitHub users"
    echo ""
    echo "🎯 Next steps:"
    echo "1. Share your repository with others"
    echo "2. Add topics/tags on GitHub (ros2, computer-vision, drone, yolo, object-detection)"
    echo "3. Consider adding a license file"
    echo "4. Update README.md with any specific instructions"
    echo ""
    echo "Happy coding! 🚁📱"
else
    echo ""
    echo "❌ Failed to push to GitHub"
    echo "=========================="
    echo ""
    echo "Possible solutions:"
    echo "1. Check your GitHub credentials"
    echo "2. Make sure the repository exists on GitHub"
    echo "3. Try using SSH instead of HTTPS:"
    echo "   git remote set-url origin git@github.com:${github_username}/${repo_name}.git"
    echo "4. Check your internet connection"
    echo ""
    echo "You can also push manually:"
    echo "git push -u origin master"
fi
