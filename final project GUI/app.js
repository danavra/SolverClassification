
// function solve_problem() {
//     server = "http://127.0.0.1:5000"
//     func = '/solve_problem'
//
//     toggleNotification('Processing in progress', 'The magic is happening')
//     $.ajax({
//         type: "POST",
//         url: server + func,
//         data: JSON.stringify(inputs),
//         dataType: 'json',
//         success: function (data) {
//             $('#toggle_modal').modal('hide')
//             popNotification(JSON.stringify(data), 'It Fucking Works!!', 'green')
//
//         }, error: function (data) {
//             $('#toggle_modal').modal('hide')
//             popError(data, "OH SHIT!")
//         }
//     })
// }

function solve_problem() {
    server = "http://127.0.0.1:5000"
    func = '/solve_problem'

    // validate input
    problem_file = document.getElementById('new_problem')
    if(problem_file.value == null || problem_file.value === ''){
        popError('Must upload a problem', 'Bad Input','left')
        return
    }

    $('#solve_problem').modal('hide')
    var fileInput = document.getElementById('new_problem');
    var file = fileInput.files[0];
    var form_data = new FormData();
    form_data.append('file', file);
    $.ajax({
        
        url:  server + func,
		dataType: 'text',  // what to expect back from the PHP script, if anything
		cache: false,
		contentType: false,
		processData: false,
		data: form_data,
		type: 'post',
        success: function (data) {
            $('#toggle_modal').modal('hide')
            popNotification(JSON.stringify(data), 'It Fucking Works!!', 'green')

        }, error: function (data) {
            $('#toggle_modal').modal('hide')
            popError('Connection refused', "OH SHIT!")
        }
    })
}


function db_analasys() {
    server = "http://127.0.0.1:5000"
    func = '/db_analasys'
    selected = {}

    $('#db_analasys').modal('hide')


    // gather users input
    $('#action_selection input').each(function() {
        if($(this). is(":checked")){
            selected[$(this).attr('name')] = "1"
        }
        else{
            selected[$(this).attr('name')] = "0"

        }
    });

    // send request to server
    toggleNotification('Analysing DB', 'The magic is happening')
    $.ajax({
        type: "POST",
        url: server + func,
        data: JSON.stringify(selected),
        dataType: 'JSON',
        success: function (data) {
            console.log('success:')
            console.log(data)
            // try{
            //      res = JSON.parse(data)
            //                 console.log('yanivvvvv')
            //
            // }catch (e) {
            //     try{
            //
            //         res = JSON.parse(JSON.stringify(data))
            //                                         console.log('yanivvvvv')
            //
            //     }catch (e) {
            //         res = data
            //                     console.log('yanivvvvv')
            //
            //     }
            // }
            res = JSON.parse(data)

            title = "DB analasys completed successfully"
            msg = 'Results:<br><br>'
            path = res['path']
            msg += 'A new file has been created at: ' + path + '<br><br>'
            for(key in res){
                if(key != 'path'){
                    msg += key + ":   " + parseInt( parseFloat(res[key])*100) + '%<br><br>'
                }
            }
            $('#toggle_modal').modal('hide')
            popNotification(msg, title,'green')

        }, error: function (data) {
            console.log(data)
            popError('something went wrong','server error')
        }
    })
}


function create_db(){
    $('#create_db').modal('hide')
    server = "http://127.0.0.1:5000"
    func = '/create_db'

    toggleNotification('Creating DataBase', 'The magic is happening')
    $.ajax({
        type: "POST",
        url: server + func,
        success: function (data) {
            $('#toggle_modal').modal('hide')
            popNotification(JSON.stringify(data), 'It Fucking Works!!', 'green')

        }, error: function (data) {
            $('#toggle_modal').modal('hide')
            popError('Connection refused', "OH SHIT!")
        }
    })
}


// function get_inputs(){
//     ans = {}
//     ans['new_problem'] = document.getElementById('new_problem').value
//     ans['db_dir'] = document.getElementById('db_dir').value
//     ans['action'] = $("#action_selection input[type='radio']:checked").val();
//     ans['rebuild_db'] = document.getElementById('rebuild_db_cb').checked
//     return ans
// }

// function validate_inputs(inputs) {
//     ans = []
//     i = 1;
//     jQuery.each(inputs, function (field, val) { // add unique fields
//         if (val == null || val === '') {
//             console.log('field: ' + field + '  val: ' + val)
//             ans.push(i + '. ' + field)
//             i = i + 1
//         }
//     });
//     return ans;
// }




// ######################################################################################################
// ######################################################################################################
// ######################################################################################################
// ######################################################################################################
// ######################################################################################################
// ######################################################################################################
// ######################################################################################################
// ######################################################################################################
// ######################################################################################################
// ######################################################################################################
// ######################################################################################################
// ######################################################################################################


$('#new_problem').change(function () {
    console.log(this.files[0]);
});
// tooltip settings
$(function () {
    $('[data-toggle="tooltip"]').tooltip({ 'delay': { 'hide': 50 } })
})
// open modal on page load
// $(function () {
//     $('#solve_problem').modal('show')
// })

// show loading modal
function toggleNotification(header, msg = 'Loading...') {
    document.getElementById("toggle_modal_body").innerHTML = msg;
    document.getElementById("toggle_modal_header").innerHTML = header;
    document.getElementById("toggle_modal_header").style.color = "black";
    document.getElementById("loading_icon").style.display = "block";
    $("#toggle_modal").modal("toggle");
}

// ######################################################################################################
// # Name:         popError
// # Description:  pops the error modal (red title)
// # Variables:    msg: the body of the modal
// #               title: the title of the modal (default="ERROR")
// ######################################################################################################
function popError(msg, title = "ERROR", text_allignment = 'center') {
    document.getElementById('alert_body').innerHTML = msg
    document.getElementById('alert_body').style.textAlign = text_allignment
    document.getElementById('alert_title').innerHTML = title
    $('#alert').modal('show')
}

// ######################################################################################################
// # Name:         popNotification
// # Description:  pops the notification modal
// # Variables:    msg: the body of the modal
// #               title: the title of the modal
// #               t_color: title color (default black)
// #               t_align: text alignment of the body
// ######################################################################################################
function popNotification(msg, title_txt, t_color = 'black', t_align = 'left') {
    body = document.getElementById('notification_body')
    title = document.getElementById('notification_title')

    body.innerHTML = msg
    body.style.textAlign = t_align
    body.style.color = 'black'
    title.innerHTML = title_txt
    title.style.color = t_color


    $('#notification_modal').modal('show')
}