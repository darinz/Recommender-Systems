#!/usr/bin/env python3
"""
Demo: Content-Based Movie Recommender System
============================================

This script demonstrates the updated movie recommender system with:
- Proper module imports
- Comprehensive inline comments
- Robust error handling
- Educational explanations
- Visualization capabilities

Run this script to see the recommender system in action!
"""

import sys
import warnings
warnings.filterwarnings('ignore')

# Import the updated movie recommender
from movie_recommender import main

if __name__ == "__main__":
    print("🎬 MOVIE RECOMMENDER SYSTEM DEMO")
    print("=" * 50)
    print("This demo will show you how content-based recommendation works!")
    print("=" * 50)
    
    try:
        # Run the main demonstration
        main()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("Please check that all dependencies are installed:")
        print("pip3 install -r requirements.txt")
        sys.exit(1) 