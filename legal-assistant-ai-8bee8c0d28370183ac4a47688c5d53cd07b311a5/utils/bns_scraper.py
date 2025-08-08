import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from tqdm import tqdm
import time

def get_bns_sections():
    """Scrape BNS sections from devgan.in"""
    base_url = "https://devgan.in/all/bareact/bharatiya_nyaya_sanhita_2023/"
    
    try:
        # Fetch the main page
        print("Fetching BNS sections from devgan.in...")
        response = requests.get(base_url)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all section links
        section_links = []
        for link in soup.select('a[href^="section/"]'):
            section_links.append({
                'section': link.text.strip(),
                'url': base_url + link['href']
            })
        
        # Collect section details
        sections = []
        for item in tqdm(section_links, desc="Scraping section details"):
            try:
                # Fetch section page
                section_response = requests.get(item['url'])
                section_response.raise_for_status()
                
                # Parse section page
                section_soup = BeautifulSoup(section_response.text, 'html.parser')
                
                # Extract section details
                title = section_soup.find('h1').text.strip()
                content_div = section_soup.find('div', {'class': 'content'})
                
                if content_div:
                    # Remove script and style elements
                    for script in content_div(['script', 'style']):
                        script.decompose()
                    
                    # Get clean text content
                    content = content_div.get_text('\n', strip=True)
                    
                    # Split into description and punishment if possible
                    parts = re.split(r'Punishment(?:—|—|--|:|-|—|—|--|:|-|\n)', content, maxsplit=1, flags=re.IGNORECASE)
                    
                    description = parts[0].strip()
                    punishment = parts[1].strip() if len(parts) > 1 else "Not specified"
                    
                    # Clean up the text
                    description = re.sub(r'\s+', ' ', description).strip()
                    punishment = re.sub(r'\s+', ' ', punishment).strip()
                    
                    sections.append({
                        'section': item['section'],
                        'title': title.replace(item['section'], '').strip(' -'),
                        'description': description,
                        'punishment': punishment,
                        'url': item['url']
                    })
                
                # Be polite with requests
                time.sleep(1)
                
            except Exception as e:
                print(f"Error scraping {item['url']}: {str(e)}")
                continue
        
        return sections
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return []

def save_to_csv(sections, filename='bns_sections.csv'):
    """Save scraped sections to a CSV file"""
    if sections:
        df = pd.DataFrame(sections)
        df.to_csv(filename, index=False)
        print(f"\nSaved {len(sections)} BNS sections to {filename}")
        return True
    return False

if __name__ == "__main__":
    # Scrape BNS sections
    bns_sections = get_bns_sections()
    
    # Save to CSV
    if bns_sections:
        save_to_csv(bns_sections, 'data/raw/bns_sections.csv')
    else:
        print("No sections were scraped. Using fallback data.")
        # Create a minimal fallback dataset
        fallback_data = [
            {
                'section': '101',
                'title': 'Punishment for murder',
                'description': 'Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine.',
                'punishment': 'Death or imprisonment for life and fine',
                'url': 'https://devgan.in/all/bareact/bharatiya_nyaya_sanhita_2023/section/101/'
            },
            {
                'section': '104',
                'title': 'Causing death by negligence',
                'description': 'Whoever causes the death of any person by doing any rash or negligent act not amounting to culpable homicide shall be punished with imprisonment of either description for a term which may extend to two years, or with fine, or with both.',
                'punishment': 'Imprisonment up to 2 years, or fine, or both',
                'url': 'https://devgan.in/all/bareact/bharatiya_nyaya_sanhita_2023/section/104/'
            }
        ]
        save_to_csv(fallback_data, 'data/raw/bns_sections_fallback.csv')
