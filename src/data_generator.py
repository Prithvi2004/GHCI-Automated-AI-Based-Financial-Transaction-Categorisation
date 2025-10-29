"""
Advanced synthetic transaction data generator
Generates realistic financial transaction data with variations, noise, and edge cases
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import yaml

class TransactionDataGenerator:
    def __init__(self, config_path="config/categories.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extended merchant templates for each category
        self.merchant_templates = {
            "Shopping": [
                "AMAZON.COM*{}", "TARGET STORE #{}", "WALMART SUPER #{}", 
                "BEST BUY #{}", "EBAY INC", "APPLE.COM/BILL", "COSTCO WHSE #{}",
                "MACYS #{}", "NORDSTROM #{}", "IKEA STORE", "HOME DEPOT #{}",
                "LOWES #{}", "KOHLS #{}", "TJ MAXX #{}", "ROSS STORES"
            ],
            "Coffee/Dining": [
                "STARBUCKS #{}", "DUNKIN ##{}", "PEETS COFFEE", "BLUE BOTTLE",
                "MCDONALDS F{}", "CHIPOTLE #{}", "PANERA BREAD #{}",
                "SUBWAY #{}", "TACO BELL #{}", "WENDYS #{}", "CHICK-FIL-A #{}",
                "OLIVE GARDEN #{}", "CHILIS #{}", "APPLEBEES #{}", "CAFE COFFEE DAY"
            ],
            "Fuel": [
                "SHELL OIL #{}", "EXXON MOBIL", "CHEVRON #{}", "BP GAS STATION",
                "SUNOCO #{}", "MOBIL GAS", "ARCO AMPM", "VALERO #{}", 
                "TEXACO #{}", "MARATHON PETRO", "CIRCLE K", "7-ELEVEN #{}",
                "SHELL - FUEL ONLY", "CHEVRON GAS STN"
            ],
            "Transport": [
                "UBER * TRIP", "LYFT * RIDE", "UBER * EATS", "METRO TRANSIT",
                "AMTRAK", "GREYHOUND", "TAXI CAB CO", "PARKING METER",
                "BART STATION", "NYC SUBWAY", "UBER TRIP {}", "LYFT RIDE {}",
                "YELLOWCAB", "TOLL ROAD", "AIRPORT PARKING"
            ],
            "Groceries": [
                "KROGER GROCERY", "WHOLE FOODS MKT", "SAFEWAY STORE #{}",
                "ALDI #{}", "TRADER JOES", "PUBLIX SUPER", "WEGMANS #{}",
                "HARRIS TEETER", "GIANT FOOD", "FOOD LION #{}", "SPROUTS #{}",
                "FRESH MARKET", "RALPHS #{}", "VONS STORE", "ALBERTSONS"
            ],
            "Health": [
                "CVS PHARMACY #{}", "WALGREENS #{}", "RITE AID PHARMACY",
                "URGENT CARE CLINIC", "DR {} OFFICE", "DENTAL CARE",
                "LABCORP", "QUEST DIAGNOSTICS", "MINUTE CLINIC",
                "OPTOMETRY CENTER", "PHYSICAL THERAPY", "CHIROPRACTOR"
            ],
            "Entertainment": [
                "NETFLIX.COM", "SPOTIFY PREMIUM", "HULU", "DISNEY PLUS",
                "HBO MAX", "AMAZON PRIME VIDEO", "YOUTUBE PREMIUM",
                "APPLE MUSIC", "XBOX GAME PASS", "PLAYSTATION STORE",
                "STEAM GAMES", "AMC THEATRES #{}", "REGAL CINEMAS",
                "TICKETMASTER", "STUBHUB", "EVENTBRITE"
            ],
            "Transfers": [
                "ZELLE TO {}", "PAYPAL TRANSFER", "VENMO PAYMENT", 
                "WIRE TRANSFER", "ACH TRANSFER", "CASH APP", "PAYPAL *{}",
                "ZELLE FROM {}", "SQUARE CASH", "APPLE PAY CASH", "GOOGLE PAY"
            ],
            "Cash": [
                "ATM WITHDRAWAL", "CASH ADVANCE", "ATM CASH DISPENSE",
                "BANK OF AMERICA ATM", "CHASE ATM", "WELLS FARGO ATM",
                "ATM FEE", "CASH BACK", "ATM DEBIT"
            ],
            "Food Delivery": [
                "DOORDASH*{}", "GRUBHUB*{}", "UBER EATS", "POSTMATES",
                "SEAMLESS", "CAVIAR", "INSTACART", "GOPUFF", 
                "DELIVEROO", "JUST EAT"
            ],
            "Utilities": [
                "PG&E ELECTRIC", "WATER DEPT", "GAS COMPANY", "COMCAST CABLE",
                "AT&T INTERNET", "VERIZON WIRELESS", "T-MOBILE", "SPRINT",
                "CON EDISON", "WASTE MANAGEMENT", "REPUBLIC SERVICES"
            ],
            "Insurance": [
                "GEICO INSURANCE", "STATE FARM", "PROGRESSIVE", "ALLSTATE",
                "USAA", "BLUE CROSS", "CIGNA", "AETNA", "UNITED HEALTHCARE"
            ]
        }
        
        # Amount ranges for each category
        self.amount_ranges = {
            "Shopping": (10, 500),
            "Coffee/Dining": (5, 150),
            "Fuel": (20, 100),
            "Transport": (8, 80),
            "Groceries": (25, 300),
            "Health": (15, 500),
            "Entertainment": (8, 200),
            "Transfers": (50, 5000),
            "Cash": (20, 500),
            "Food Delivery": (15, 80),
            "Utilities": (50, 400),
            "Insurance": (100, 600)
        }
        
        # Name pool for transfers
        self.names = [
            "JOHN SMITH", "SARAH JOHNSON", "MIKE CHEN", "EMILY DAVIS",
            "ALEX WILLIAMS", "JESSICA BROWN", "DAVID LEE", "MARIA GARCIA"
        ]
    
    def generate_description(self, category, add_noise=False):
        """Generate a transaction description for given category"""
        templates = self.merchant_templates.get(category, ["UNKNOWN MERCHANT"])
        template = random.choice(templates)
        
        # Fill in template placeholders
        if "{}" in template:
            if "TO" in template or "FROM" in template:
                desc = template.format(random.choice(self.names))
            else:
                desc = template.format(random.randint(100, 9999))
        else:
            desc = template
        
        # Add noise variations
        if add_noise:
            noise_type = random.choice(['typo', 'extra', 'truncate', 'case', 'none', 'none', 'none'])
            if noise_type == 'typo':
                # Introduce typo
                desc = self._add_typo(desc)
            elif noise_type == 'extra':
                # Add extra transaction ID or reference
                desc += f" REF#{random.randint(1000, 9999)}"
            elif noise_type == 'truncate':
                # Truncate description
                if len(desc) > 10:
                    desc = desc[:random.randint(8, len(desc)-2)]
            elif noise_type == 'case':
                # Random case
                desc = desc.lower() if random.random() > 0.5 else desc.upper()
        
        return desc
    
    def _add_typo(self, text):
        """Add a random typo to text"""
        if len(text) < 3:
            return text
        pos = random.randint(1, len(text) - 2)
        chars = list(text)
        chars[pos] = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        return ''.join(chars)
    
    def generate_dataset(self, num_samples=1200, noise_prob=0.15, train_test_split=0.8):
        """
        Generate synthetic transaction dataset
        
        Args:
            num_samples: Total number of transactions to generate
            noise_prob: Probability of adding noise to descriptions
            train_test_split: Train/test split ratio
        
        Returns:
            train_df, test_df
        """
        data = []
        
        # Get all categories
        categories = list(self.merchant_templates.keys())
        
        # Generate balanced dataset with some variation
        samples_per_category = num_samples // len(categories)
        
        for category in categories:
            # Add some variation (±20%) to balance
            n_samples = int(samples_per_category * random.uniform(0.8, 1.2))
            
            for _ in range(n_samples):
                add_noise = random.random() < noise_prob
                
                description = self.generate_description(category, add_noise)
                amount = round(random.uniform(*self.amount_ranges[category]), 2)
                
                # Generate random date within last year
                days_ago = random.randint(0, 365)
                date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
                
                # Transaction type
                txn_type = random.choice(['debit', 'credit']) if category != "Transfers" else random.choice(['debit', 'credit', 'transfer'])
                
                data.append({
                    'description': description,
                    'amount': amount,
                    'date': date,
                    'type': txn_type,
                    'category': category
                })
        
        # Create DataFrame and shuffle
        df = pd.DataFrame(data)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split train/test
        split_idx = int(len(df) * train_test_split)
        train_df = df[:split_idx]
        test_df = df[split_idx:]
        
        return train_df, test_df
    
    def save_datasets(self, output_dir="data"):
        """Generate and save train/test datasets"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("🔄 Generating synthetic transaction data...")
        train_df, test_df = self.generate_dataset(num_samples=1200, noise_prob=0.15)
        
        train_path = f"{output_dir}/train_transactions.csv"
        test_path = f"{output_dir}/test_transactions.csv"
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"✅ Generated {len(train_df)} training samples → {train_path}")
        print(f"✅ Generated {len(test_df)} test samples → {test_path}")
        print(f"📊 Categories: {train_df['category'].nunique()}")
        print(f"📊 Category distribution:")
        print(train_df['category'].value_counts().to_string())
        
        return train_df, test_df


if __name__ == "__main__":
    generator = TransactionDataGenerator()
    generator.save_datasets()
