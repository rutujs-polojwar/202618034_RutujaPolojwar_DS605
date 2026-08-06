import scrapy


class BooksSpider(scrapy.Spider):
    name = "books"
    allowed_domains = ["books.toscrape.com"]
    for i in range(1 ,6) : 
        start_urls = [
            f"https://books.toscrape.com/catalogue/page-{i}.html",
        ]

    def parse(self, response):

        # Find all books on the current catalogue page
        books = response.css("article.product_pod")

        # Visit each book page
        for book in books:
            book_url = book.css("h3 a::attr(href)").get()

            yield response.follow(
                book_url,
                callback=self.parse_book
            )
        

    def parse_book(self, response):

        # Basic details
        title = response.css("div.product_main h1::text").get()

        price = response.css("p.price_color::text").get()

        availability = response.css("p.instock.availability").xpath("normalize-space()").get()

        rating = response.css(
            "p.star-rating::attr(class)"
        ).get().split()[1]

        category = response.css(
            "ul.breadcrumb li a::text"
        ).getall()[-1]

        description = response.css(
            "#product_description + p::text"
        ).get()

        # Product information table
        table = {}

        for row in response.css("table tr"):

            key = row.css("th::text").get()

            value = row.css("td::text").get()

            table[key] = value

        upc = table.get("UPC")

        reviews = table.get("Number of reviews")

        # Save one book
        yield {
            "title": title,
            "category": category,
            "price": price,
            "rating": rating,
            "availability": availability,
            "description": description,
            "UPC": upc,
            "reviews": reviews,
            "url": response.url,
        }