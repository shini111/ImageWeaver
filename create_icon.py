#!/usr/bin/env python3
"""
Create a simple icon for ImageWeaver
"""

try:
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a 256x256 image with transparent background
    size = 256
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a blue circle background
    margin = 20
    draw.ellipse([margin, margin, size-margin, size-margin], 
                fill=(52, 152, 219, 255), outline=(41, 128, 185, 255), width=4)
    
    # Draw the "IW" text in white
    try:
        # Try to use a nice font
        font = ImageFont.truetype("arial.ttf", 80)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    text = "IW"
    
    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the text
    x = (size - text_width) // 2
    y = (size - text_height) // 2 - 10  # Slightly higher
    
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
    
    # Save as ICO file
    img.save('icon.ico', format='ICO', sizes=[(256, 256), (128, 128), (64, 64), (32, 32), (16, 16)])
    print("✅ Icon created: icon.ico")
    
except ImportError:
    print("⚠️ PIL not available. Creating a simple placeholder icon...")
    
    # Create a minimal ICO file without PIL
    # This is a very basic approach - just create a small file
    ico_data = b'\x00\x00\x01\x00\x01\x00\x10\x10\x00\x00\x01\x00\x20\x00\x68\x04\x00\x00\x16\x00\x00\x00'
    ico_data += b'\x00' * 1128  # Minimal ICO structure
    
    with open('icon.ico', 'wb') as f:
        f.write(ico_data)
    
    print("✅ Basic icon created: icon.ico")

except Exception as e:
    print(f"❌ Failed to create icon: {e}")
    print("Continuing without icon...")
