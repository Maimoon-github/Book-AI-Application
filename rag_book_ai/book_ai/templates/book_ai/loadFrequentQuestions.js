// Function to load frequent questions for a chapter
function loadFrequentQuestions(chapterId) {
    // If no chapter is selected, clear the questions area
    if (!chapterId) {
        $('#frequent-questions').html('');
        return;
    }
    
    // Add loading spinner
    $('#frequent-questions').html(`
        <div class="text-center py-2">
            <div class="spinner-border spinner-border-sm text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <small class="text-muted ms-2">Loading questions...</small>
        </div>
    `);
    
    // Fetch frequent questions from server
    $.ajax({
        url: `/chapter/${chapterId}/frequent-questions/`,
        type: 'GET',
        success: function(response) {
            if (response.questions && response.questions.length > 0) {
                let questionsHtml = `
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <h6 class="mb-0">
                            <i class="fas fa-fire text-warning me-1"></i>Popular Questions
                        </h6>
                        <small class="text-muted">${response.questions.length} question${response.questions.length > 1 ? 's' : ''}</small>
                    </div>
                    <div class="list-group list-group-flush popular-questions">
                `;
                
                response.questions.forEach(function(q) {
                    questionsHtml += `
                        <button class="list-group-item list-group-item-action border-0 py-2 px-3 popular-question"
                                onclick="insertSuggestion('${q.text.replace(/'/g, "\\'")}')">
                            <div class="d-flex align-items-center">
                                <i class="fas fa-question-circle text-primary me-2"></i>
                                <span>${q.text}</span>
                            </div>
                            <small class="badge bg-light text-dark mt-1">${q.frequency} ${q.frequency > 1 ? 'times' : 'time'}</small>
                        </button>
                    `;
                });
                
                questionsHtml += '</div>';
                $('#frequent-questions').html(questionsHtml);
                
                // Update question count
                $('#question-count').text(`${response.questions.length} question${response.questions.length > 1 ? 's' : ''}`);
            } else {
                $('#frequent-questions').html(`
                    <div class="text-center py-3 border-top mt-2">
                        <i class="fas fa-question-circle text-muted mb-2" style="font-size: 1.5rem;"></i>
                        <p class="mb-0 small text-muted">No questions yet for this chapter</p>
                    </div>
                `);
                $('#question-count').text('0 questions');
            }
        },
        error: function() {
            $('#frequent-questions').html(`
                <div class="text-center py-2">
                    <i class="fas fa-exclamation-circle text-warning mb-2"></i>
                    <p class="mb-0 small text-muted">Couldn't load questions</p>
                </div>
            `);
            $('#question-count').text('');
        }
    });
}
