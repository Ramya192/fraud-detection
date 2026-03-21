from PIL import Image, ImageDraw, ImageFont
import os

def create_test_receipt():
    """Create a fake bank receipt image for testing"""

    # Create white image
    img= Image.new('RGB',(400,300),color = 'white')
    draw= ImageDraw.Draw(img)

    # Draw Receipt Content
    draw.rectangle([20,20,380,380],outline= 'black',width=2)

    # Receipt text
    lines = [
        "BANK RECEIPT",
        "=" *30,
        "Date: 2026-03-20",
        "Time: 02:34 AM",
        "Merchant: Unknown Store",
        "Location: Mumbai",
        "",
        "AMOUNT: $1,250.00",
        "",
        "Transaction ID: TXN9981",
        "Status: PENDING",
        "=" * 30,
        "Keep this receipt"
    ]

    y=40
    for line in lines:
        draw.text((40,y),line,fill='black')
        y += 18

    # Save image
    os.makedirs('data/test_images',exist_ok=True)
    img.save('data/test_images/test_receipt.jpg')
    print("Test receipt created: data/test_images/test_receipt.jpg")

if __name__ == "__main__":
    create_test_receipt()