// Book Dashboard Enhanced Features
$(document).ready(function() {
    // Reading progress sliders
    $('input[type="range"][name="current_page"]').on('input', function() {
        var bookId = $(this).closest('form').data('book-id');
        $('#currentPageDisplay' + bookId).text($(this).val());
    });
    
    // Toggle favorite status
    $('.toggle-favorite').on('click', function() {
        var btn = $(this);
        var bookId = btn.data('book-id');
        var currentFavorite = btn.data('favorite');
        
        $.ajax({
            url: `/book/${bookId}/update/`,
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                favorite: !currentFavorite
            }),
            success: function(response) {
                if (response.success) {
                    btn.data('favorite', !currentFavorite);
                    if (!currentFavorite) {
                        btn.find('i').removeClass('fa-heart-broken').addClass('fa-heart');
                    } else {
                        btn.find('i').removeClass('fa-heart').addClass('fa-heart-broken');
                    }
                }
            }
        });
    });
    
    // Save book changes from modal
    $('.save-book-changes').on('click', function() {
        var bookId = $(this).data('book-id');
        var form = $('#editBookForm' + bookId);
        
        var bookData = {
            title: form.find('[name="title"]').val(),
            author: form.find('[name="author"]').val(),
            category: form.find('[name="category"]').val(),
            tags: form.find('[name="tags"]').val(),
            description: form.find('[name="description"]').val(),
            current_page: parseInt(form.find('[name="current_page"]').val()),
            favorite: form.find('[name="favorite"]').is(':checked')
        };
        
        $.ajax({
            url: `/book/${bookId}/update/`,
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(bookData),
            success: function(response) {
                if (response.success) {
                    $('#editBookModal' + bookId).modal('hide');
                    location.reload(); // Refresh to show updates
                }
            }
        });
    });
    
    // Update reading progress
    $('input[type="range"][name="current_page"]').on('change', function() {
        var bookId = $(this).closest('form').data('book-id');
        var currentPage = parseInt($(this).val());
        
        $.ajax({
            url: `/book/${bookId}/progress/`,
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                current_page: currentPage
            }),
            success: function(data) {
                if (data.success) {
                    // Update progress percentage display
                    var progressPercentage = data.progress_percentage;
                    $(`#progress-${bookId}`).css('width', progressPercentage + '%');
                    $(`#progress-${bookId}`).attr('aria-valuenow', progressPercentage);
                    $(`#progress-${bookId}`).text(progressPercentage + '%');
                }
            }
        });
    });
    
    // Book recommendations
    function loadRecommendations() {
        $.ajax({
            url: '/api/recommendations/',
            method: 'GET',
            success: function(data) {
                var recommendationsHtml = '';
                if (data.recommendations && data.recommendations.length > 0) {
                    for (var i = 0; i < data.recommendations.length; i++) {
                        var book = data.recommendations[i];
                        recommendationsHtml += `
                            <div class="col-md-4">
                                <div class="card h-100">
                                    <div class="card-body">
                                        <h5 class="card-title">${book.title}</h5>
                                        <p class="card-text">
                                            <small class="text-muted">
                                                <i class="fas fa-user"></i> ${book.author || 'Unknown author'}<br>
                                                <i class="fas fa-tag"></i> ${book.category || 'Uncategorized'}
                                            </small>
                                        </p>
                                        <a href="/book/${book.id}/" class="btn btn-outline-primary btn-sm">
                                            <i class="fas fa-eye"></i> View Details
                                        </a>
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                } else {
                    recommendationsHtml = '<div class="col-12 text-center"><p class="text-muted">No recommendations available.</p></div>';
                }
                $('#book-recommendations').html(recommendationsHtml);
            },
            error: function() {
                $('#book-recommendations').html('<div class="col-12 text-center"><p class="text-danger">Failed to load recommendations.</p></div>');
            }
        });
    }
    
    // Load recommendations on page load
    if ($('#book-recommendations').length) {
        loadRecommendations();
    }
    
    // Update book detail page reading progress
    $('#updateProgressBtn').on('click', function() {
        var bookId = $(this).data('book-id');
        var currentPage = parseInt($('#currentPageInput').val());
        
        $.ajax({
            url: `/book/${bookId}/progress/`,
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                current_page: currentPage
            }),
            success: function(data) {
                if (data.success) {
                    // Update progress bar
                    $('.progress-bar').css('width', data.progress_percentage + '%');
                    $('.progress-bar').attr('aria-valuenow', data.progress_percentage);
                    $('.progress-bar').text(data.progress_percentage + '%');
                    
                    // Show success message using bootstrap toast
                    $('.toast').toast('show');
                }
            }
        });
    });
});
